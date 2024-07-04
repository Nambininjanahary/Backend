import logging
from pydub import AudioSegment
import numpy as np
import torch
import requests
from transformers import pipeline,T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizerFast, EncoderDecoderModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import pipeline as transformers_pipeline
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import AudioFile
from .serializers import AudioFileSerializer
from faster_whisper import WhisperModel
import spacy
import pytextrank
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def read_audio(file_path, target_sample_rate=16000):
    try:
        audio = AudioSegment.from_file(file_path)
        if audio.frame_rate != target_sample_rate:
            logger.debug(f"Converting audio from {audio.frame_rate} Hz to {target_sample_rate} Hz")
            audio = audio.set_frame_rate(target_sample_rate)
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
        return audio_array, audio.frame_rate
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        raise

class TranscriptionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    def post(self, request, *args, **kwargs):
        logger.debug("Received transcription request: %s", request.data)
        file_serializer = AudioFileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            audio_file = file_serializer.instance

            try:
                audio_path = audio_file.file.path
                logger.debug(f"Audio file path: {audio_path}")

                audio_array, sample_rate = read_audio(audio_path)
                logger.debug(f"Audio array length: {len(audio_array)}, Sample rate: {sample_rate}")

                transcription_result = self.transcribe_audio(audio_path)
                logger.debug(f"Transcription: {transcription_result}")

                audio_file.transcription = transcription_result
                audio_file.save()

                response_data = file_serializer.data
                response_data['transcription'] = transcription_result

                logger.debug(f"Response data: {response_data}")
                return Response(response_data, status=201)
            except Exception as e:
                logger.error(f"Error processing transcription request: {e}")
                return Response({'error': str(e)}, status=500)
        else:
            logger.debug(f"File serializer errors: {file_serializer.errors}")
            return Response(file_serializer.errors, status=400)

    def transcribe_audio(self, audio_path):
        logger.debug(f"Transcribing audio file: {audio_path}")
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        return transcription
    
class SummarizationView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = 'cpu'        
        self.tokenizer = T5Tokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
        self.model = T5ForConditionalGeneration.from_pretrained("plguillou/t5-base-fr-sum-cnndm")

    def post(self, request, *args, **kwargs):
        logger.debug("Received summarization request: %s", request.data)
        transcription = request.data.get('transcription', '')
        if not transcription:
            return Response({'error': 'Transcription is required for summarization'}, status=400)

        try:
            summary_text = self.summarize_text(transcription)
            logger.debug(f"Summary: {summary_text}")
            return Response({'summary': summary_text}, status=200)
        except Exception as e:
            logger.error(f"Error processing summarization request: {e}")
            return Response({'error': str(e)}, status=500)

    def summarize_text(self, text):
        logger.debug(f"Summarizing text")
        inputs = self.tokenizer([text], padding="max_length", truncation=True, max_length=258, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1024)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)