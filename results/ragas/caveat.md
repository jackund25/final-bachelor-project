# Caveat LLM-as-judge (RAGAS)

Skor RAGAS dihasilkan Gemini (LLM) yang menilai output LLM → berpotensi bias self-referential dan bervariasi antar-run. context_recall/precision (berbasis reference) lebih objektif daripada faithfulness/answer_relevancy. Angka bersifat indikatif.

Judge: gemini-2.5-flash-lite | Generator: gemini-2.5-flash | n_kasus: 4 | kasus: ['D1', 'D2', 'D4', 'D5'] | top_k: 4
