"""
LLM Generator for RAG explanations
"""

import os
from typing import Dict, List, Optional
import logging

# Try different LLM providers
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq not available. Install: pip install groq")

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    logging.warning("llama-cpp-python not available for local LLM")

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    Generate explanations using LLM with retrieved context
    """
    
    def __init__(self, provider: str = "groq", model_config: Optional[Dict] = None):
        """
        Initialize generator
        
        Args:
            provider: LLM provider ('groq', 'local', or 'template')
            model_config: Configuration for the model
        """
        self.provider = provider
        self.config = model_config or {}
        
        if provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq not installed. Install: pip install groq")
            
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            
            self.client = Groq(api_key=api_key)
            self.model_name = self.config.get('model', 'llama-3.1-70b-versatile')
            
            logger.info(f"Initialized Groq client with model {self.model_name}")
        
        elif provider == "local":
            if not LLAMACPP_AVAILABLE:
                raise ImportError("llama-cpp-python not installed")
            
            model_path = self.config.get('model_path')
            if not model_path:
                raise ValueError("model_path required for local LLM")
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.config.get('context_window', 2048),
                n_threads=self.config.get('threads', 4),
                n_gpu_layers=self.config.get('gpu_layers', 0)
            )
            
            logger.info(f"Initialized local LLM from {model_path}")
        
        elif provider == "template":
            logger.info("Using template-based generator (no LLM)")
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate_explanation(self, 
                           context_docs: List[str],
                           patient_state: Dict,
                           prediction: float,
                           temperature: float = 0.7,
                           max_tokens: int = 300) -> str:
        """
        Generate clinical explanation
        
        Args:
            context_docs: Retrieved medical guideline texts
            patient_state: Current patient state
            prediction: Predicted glucose value
            temperature: LLM temperature
            max_tokens: Maximum response length
            
        Returns:
            explanation: Generated explanation text
        """
        # Build prompt
        prompt = self._build_prompt(context_docs, patient_state, prediction)
        
        # Generate based on provider
        if self.provider == "groq":
            explanation = self._generate_groq(prompt, temperature, max_tokens)
        
        elif self.provider == "local":
            explanation = self._generate_local(prompt, temperature, max_tokens)
        
        elif self.provider == "template":
            explanation = self._generate_template(patient_state, prediction)
        
        else:
            explanation = "Generator not configured properly."
        
        return explanation
    
    def _build_prompt(self, context_docs: List[str], 
                     patient_state: Dict, prediction: float) -> str:
        """
        Build prompt for LLM
        
        Args:
            context_docs: Retrieved documents
            patient_state: Patient state
            prediction: Prediction value
            
        Returns:
            prompt: Formatted prompt
        """
        context = "\n\n".join(context_docs)
        
        prompt = f"""Anda adalah asisten kesehatan untuk pasien diabetes di Indonesia. Berikan penjelasan yang akurat, mudah dimengerti, dan empati.

PANDUAN MEDIS:
{context}

DATA PASIEN:
- Gula darah sekarang: {patient_state.get('current_glucose', 'N/A')} mg/dL
- Tingkat stress: {patient_state.get('stress_level', 'N/A')}/10
- Aktivitas fisik hari ini: {patient_state.get('activity_level', 0)} menit
- Karbohidrat aktif: {patient_state.get('carbs_on_board', 0)} gram
- Insulin aktif: {patient_state.get('insulin_on_board', 0)} unit

PREDIKSI: Gula darah 1 jam ke depan = {prediction:.1f} mg/dL

TUGAS ANDA:
1. **Penjelasan Kausal**: Jelaskan MENGAPA prediksi ini masuk akal berdasarkan data pasien (hubungkan dengan stress, aktivitas, dll)
2. **Penilaian Risiko**: Tentukan apakah ini AMAN / HATI-HATI / BAHAYA
3. **Rekomendasi Aksi**: Berikan 2-3 langkah konkret yang bisa dilakukan pasien

ATURAN:
- Gunakan Bahasa Indonesia yang natural
- Maksimal 150 kata
- Hindari jargon medis yang rumit
- Fokus pada actionable advice
- Gunakan nada yang supportif, bukan menakut-nakuti

JAWABAN:"""

        return prompt
    
    def _generate_groq(self, prompt: str, temperature: float, 
                      max_tokens: int) -> str:
        """Generate using Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._generate_template({}, 0)  # Fallback
    
    def _generate_local(self, prompt: str, temperature: float,
                       max_tokens: int) -> str:
        """Generate using local LLM"""
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "\n\n\n"]
            )
            
            return output['choices'][0]['text']
        
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return self._generate_template({}, 0)  # Fallback
    
    def _generate_template(self, patient_state: Dict, prediction: float) -> str:
        """
        Template-based generation (Plan C fallback)
        
        Args:
            patient_state: Patient state
            prediction: Prediction
            
        Returns:
            explanation: Template-based text
        """
        current_glucose = patient_state.get('current_glucose', 100)
        stress = patient_state.get('stress_level', 5)
        activity = patient_state.get('activity_level', 0)
        
        # Determine risk
        if prediction > 180:
            risk = "HATI-HATI - Hiperglikemia"
            risk_emoji = "⚠️"
        elif prediction < 70:
            risk = "BAHAYA - Hipoglikemia"
            risk_emoji = "🚨"
        else:
            risk = "AMAN"
            risk_emoji = "✅"
        
        # Build causal explanation
        causes = []
        if stress > 6:
            causes.append(f"tingkat stress tinggi ({stress}/10)")
        if activity < 15:
            causes.append("kurangnya aktivitas fisik")
        if current_glucose > 150:
            causes.append("gula darah awal sudah tinggi")
        
        if causes:
            cause_text = f"terutama karena {', '.join(causes)}"
        else:
            cause_text = "berdasarkan kondisi metabolik saat ini"
        
        # Recommendations
        if prediction > 180:
            recommendations = """
**Rekomendasi:**
1. Perbanyak minum air putih (minimal 2 gelas)
2. Lakukan jalan santai 15-20 menit
3. Hindari makanan manis dalam 2 jam ke depan
4. Cek ulang gula darah dalam 1 jam
            """
        elif prediction < 70:
            recommendations = """
**Rekomendasi SEGERA:**
1. Konsumsi 15g karbohidrat cepat (jus/permen)
2. Duduk dan istirahat
3. Cek ulang gula darah setelah 15 menit
4. Jika masih rendah, ulangi langkah 1-3
            """
        else:
            recommendations = """
**Rekomendasi:**
1. Pertahankan pola makan seimbang
2. Lanjutkan aktivitas fisik rutin
3. Monitor gula darah berkala
            """
        
        explanation = f"""
{risk_emoji} **Status: {risk}**

**Penjelasan:**
Gula darah Anda diprediksi {'naik' if prediction > current_glucose else 'turun'} dari {current_glucose:.0f} menjadi {prediction:.0f} mg/dL {cause_text}.

{recommendations}

💡 Catatan: Prediksi ini berdasarkan pola metabolisme Anda. Selalu konsultasikan dengan dokter untuk keputusan medis penting.
        """
        
        return explanation.strip()