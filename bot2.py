import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
# ✅ Правильный импорт: WhisperSTTService использует faster-whisper бэкенд
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# ✅ FastAPI для CORS и статики
from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import importlib.resources as pkg_resources
import pipecat_ai_small_webrtc_prebuilt

load_dotenv(override=True)


async def run_bot(transport: BaseTransport):
    """Main bot logic."""
    logger.info("Starting bot")

    # ✅ Speech-to-Text: локальный Whisper (faster-whisper бэкенд)
    stt = WhisperSTTService(
        device="cpu",  # или "cuda" если есть GPU
        compute_type="int8",  # int8 для CPU, float16 для CUDA
        settings=WhisperSTTService.Settings(
            model="base",  # tiny, base, small, medium, large-v3
            language="ru",  # опционально: фиксировать язык
            no_speech_prob=0.5,
        ),
    )

    # Text-to-Speech service
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=os.getenv("KOKORO_VOICE_ID", "af_heart"),
        ),
    )

    # LLM service — Ollama на удалённом адресе
    llm = OLLamaLLMService(
        base_url="http://192.168.3.151:11434/v1",
        settings=OLLamaLLMService.Settings(
            model=os.getenv("OLLAMA_MODEL", "minimax-m2:cloud"),

            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message({"role": "user", "content": "Please introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("✅ Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("🔌 Client disconnected")
        await task.cancel()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"🗣️ Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"🤖 Transcript: {line}")

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport = None

    match runner_args:
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    # ❗ enable_audio_level_observer не влияет на клиентский JS
                ),
            )
        case DailyRunnerArguments():
            transport = DailyTransport(
                runner_args.room_url,
                runner_args.token,
                "Pipecat Bot",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    # ✅ Настраиваем CORS и UI ТОЛЬКО если есть доступ к app
    app = getattr(runner_args, 'app', None)
    
    if app:
        # CORS для разработки
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # ⚠️ В продакшене укажите конкретные домены
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("✅ CORS middleware added")

        # Монтируем prebuilt UI если пакет установлен
        try:
            dist_path = pkg_resources.files(pipecat_ai_small_webrtc_prebuilt) / "client" / "dist"
            if dist_path.exists():
                app.mount("/prebuilt", StaticFiles(directory=str(dist_path), html=True), name="prebuilt")
                
                @app.get("/", include_in_schema=False)
                async def root_redirect(request: Request):
                    return RedirectResponse(url="/prebuilt/")
                logger.info("✅ Prebuilt UI mounted at /prebuilt/")
        except Exception as e:
            logger.warning(f"⚠️ Could not mount prebuilt UI: {e}")
            logger.info("👉 Fallback: use http://<server>:7860/client/")

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()