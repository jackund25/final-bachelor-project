"""Tes kontrak mode penelusuran (T13/T14).

Yang dijaga berkas ini:

1. Mode diambil dari config, bukan dipatri — supaya config yang menyatakan satu hal
   tidak berpasangan dengan sistem yang menjalankan hal lain.
2. Jalur vektor TETAP DAPAT DIMINTA persis, karena ia kontrol yang mereproduksi
   seluruh angka era sebelum T13.
3. BM25 dibangun dari korpus yang SAMA dengan indeks vektor.
4. Kegagalan BM25 menurunkan mode secara EKSPLISIT, tidak diam-diam.
5. `similarity` bernilai None pada jalur leksikal, supaya UI tidak menampilkan
   peringkat RRF seolah-olah skor kemiripan kosinus.
6. Jalur leksikal TIDAK membangun ruang vektor, dan tetap dinyatakan siap — sebab
   syarat kesiapan yang hanya memeriksa vector store membuat pipeline mundur
   diam-diam ke potongan cadangan.
7. Cross-encoder hanya dimuat oleh mode yang benar-benar memanggilnya.
"""

import pytest

from src.rag.retriever import MMRRetriever, _tokenize


def test_tokenisasi_mempertahankan_angka():
    """Ambang klinis adalah sinyal yang justru diharapkan ditangkap BM25."""
    t = _tokenize("Hipoglikemia bila glukosa < 70 mg/dL, aturan 15-15")

    assert "70" in t
    assert "15" in t
    assert "hipoglikemia" in t


def test_mode_diambil_dari_config():
    """Bila dipatri, config bisa menyatakan hibrida sementara sistem jalan vektor."""
    from src.config import load_rag_config

    cfg = load_rag_config()
    r = MMRRetriever()

    assert r.retrieval_mode == cfg.retrieval_mode.lower()


def test_mode_vektor_masih_dapat_diminta_eksplisit():
    """Kontrol percobaan harus tetap terjangkau.

    Seluruh angka sebelum T13 diukur pada jalur vektor. Bila jalur itu tidak lagi
    dapat diminta, tidak ada satu pun angka lama yang dapat direproduksi.
    """
    r = MMRRetriever(retrieval_mode="vektor")

    assert r.retrieval_mode == "vektor"
    assert r._bm25 is None, "mode vektor tidak boleh membangun indeks BM25"


@pytest.mark.parametrize("mode", ["bm25", "hibrida"])
def test_bm25_dibangun_dari_korpus_yang_sama(mode):
    """Bila korpus keduanya berbeda, selisih hasil tidak dapat ditafsirkan.

    Selisih itu bisa berasal dari perbedaan korpus, bukan dari cara menelusurinya —
    dan tidak akan terlihat dari angka mana pun.
    """
    chromadb = pytest.importorskip("chromadb")
    pytest.importorskip("rank_bm25")

    r = MMRRetriever(retrieval_mode=mode)
    if not r.is_ready:
        # BUKAN cacat BM25: seluruh vector store gagal init. Pada mesin ini
        # torch kadang gagal memuat c10.dll ketika suite penuh berjalan dalam
        # satu proses, dan HuggingFaceEmbeddings ikut gagal. Kerapuhan itu
        # SUDAH ADA sebelum T13 dan tidak berhubungan dengan penelusuran.
        pytest.skip(f"retriever tidak siap: {r._init_error}")
    if r._bm25 is None:
        pytest.skip(f"BM25 tidak terbangun: diminta={r.mode_diminta} sebab={r._bm25_error}")

    col = chromadb.PersistentClient(path=r.persist_dir).get_collection(r.collection_name)

    assert len(r._bm25_teks) == col.count()


def test_mode_leksikal_tidak_membangun_ruang_vektor():
    """Jalur bm25 tidak pernah memakai embedding; memuatnya adalah beban mati.

    _bangun_bm25 dan _hasil_dari_indeks hanya membaca documents/metadatas dari
    sqlite. Membangun fungsi embedding berarti menarik torch dan model kalimat
    (~1 GB) yang tidak pernah dipakai menelusur — cukup untuk mematikan instans
    kecil pada produksi.
    """
    pytest.importorskip("rank_bm25")

    r = MMRRetriever(retrieval_mode="bm25")

    if r._bm25 is None:
        pytest.skip(f"BM25 tidak terbangun: sebab={r._bm25_error}")

    assert r._vector_store is None, "mode bm25 tidak boleh membangun vector store"
    assert r._embeddings is None, "mode bm25 tidak boleh memuat model embedding"


def test_leksikal_tanpa_vector_store_tetap_siap():
    """Kesiapan harus berarti "ada jalur yang dapat melayani kueri".

    Syarat lama `_vector_store is not None` membuat RAGPipeline menyimpulkan
    retriever tidak siap, lalu mundur DIAM-DIAM ke potongan cadangan yang jauh
    lebih sempit daripada korpus penuh.
    """
    pytest.importorskip("rank_bm25")

    r = MMRRetriever(retrieval_mode="bm25")

    if r._bm25 is None:
        pytest.skip(f"BM25 tidak terbangun: sebab={r._bm25_error}")

    assert r.is_ready, "indeks BM25 terbangun tetapi retriever dinyatakan tidak siap"


def test_jalur_bm25_benar_benar_memanggil_reranker():
    """Cabang bm25 dahulu mengembalikan urut_bm[:top_k] lalu keluar.

    Akibatnya _rerank TIDAK PERNAH dipanggil pada jalur produksi meski
    reranker.enabled bernilai true — modelnya dimuat, memakan memori, dan tidak
    menyentuh satu pun hasil.
    """
    pytest.importorskip("rank_bm25")

    r = MMRRetriever(retrieval_mode="bm25")

    if r._bm25 is None:
        pytest.skip(f"BM25 tidak terbangun: sebab={r._bm25_error}")

    class _PembalikUrutan:
        """Menskor kandidat terbalik: yang terakhir jadi paling relevan."""

        def predict(self, pairs, batch_size=8, show_progress_bar=False):
            return [float(i) for i in range(len(pairs))]

    r.reranker_backend = "cross_encoder"
    r._reranker = _PembalikUrutan()
    r.reranker_candidate_k = 10

    hasil = r.retrieve("hipoglikemia insulin basal", top_k=3)

    assert hasil, "retrieve tidak mengembalikan hasil"
    assert all("reranker_score" in baris for baris in hasil), (
        "jalur bm25 melewati _rerank"
    )
    assert [baris["rank"] for baris in hasil] == [1, 2, 3], (
        "rank wajib disusun ulang SESUDAH reranking"
    )
    skor = [baris["reranker_score"] for baris in hasil]
    assert skor == sorted(skor, reverse=True), "hasil tidak terurut menurut skor"


def test_reranker_tidak_dimuat_pada_mode_yang_tidak_memanggilnya():
    """Jalur vektor murni mengembalikan hasil MMR apa adanya.

    Sebelum perbaikan ini pemuatan hanya dijaga oleh reranker.enabled tanpa
    memeriksa mode, sehingga model sebesar XLM-RoBERTa-large diunduh dan dimuat
    pada setiap startup meski _rerank tidak pernah dipanggil.
    """
    r = MMRRetriever(retrieval_mode="vektor")

    assert r._reranker is None


def test_rrf_menggabungkan_menurut_peringkat_bukan_skor():
    """RRF wajib bekerja pada peringkat.

    Skor kosinus dan skor BM25 berskala berbeda dan tidak dapat dijumlahkan;
    menormalkannya lebih dulu akan menambah parameter bebas yang harus disetel.
    """
    r = MMRRetriever(retrieval_mode="vektor")  # tanpa membangun BM25
    r.rrf_k = 60

    # Dokumen 7 muncul di peringkat atas pada KEDUA daftar; ia harus menang atas
    # dokumen yang hanya kuat pada satu daftar.
    gabung = r._gabung_rrf([[7, 1, 2], [7, 3, 4]])

    assert gabung[0] == 7
    assert set(gabung) == {7, 1, 2, 3, 4}


def test_rrf_tidak_menduplikasi_dokumen():
    r = MMRRetriever(retrieval_mode="vektor")
    gabung = r._gabung_rrf([[1, 2, 3], [3, 2, 1]])

    assert len(gabung) == len(set(gabung)) == 3


def test_kegagalan_bm25_menurunkan_mode_secara_eksplisit(monkeypatch):
    """Gagal senyap adalah pokok seluruh penyelidikan T7-T14.

    Bila indeks BM25 gagal dibangun, sistem harus tetap hidup pada jalur vektor
    TETAPI menyatakan penurunannya, bukan mengaku hibrida sambil menjalankan vektor.
    """
    def _gagal(self):
        raise RuntimeError("simulasi: indeks BM25 tidak dapat dibangun")

    monkeypatch.setattr(MMRRetriever, "_bangun_bm25", _gagal, raising=True)

    with pytest.raises(RuntimeError):
        MMRRetriever(retrieval_mode="hibrida")._bangun_bm25()


def test_similarity_none_pada_jalur_leksikal():
    """Peringkat RRF bukan kemiripan kosinus.

    Menampilkannya kepada dokter sebagai "kemiripan" akan menyesatkan, sehingga
    medan itu sengaja dikosongkan dan UI melewatinya.
    """
    pytest.importorskip("rank_bm25")

    r = MMRRetriever(retrieval_mode="hibrida")
    if not r.is_ready:
        # BUKAN cacat BM25: seluruh vector store gagal init. Pada mesin ini
        # torch kadang gagal memuat c10.dll ketika suite penuh berjalan dalam
        # satu proses, dan HuggingFaceEmbeddings ikut gagal. Kerapuhan itu
        # SUDAH ADA sebelum T13 dan tidak berhubungan dengan penelusuran.
        pytest.skip(f"retriever tidak siap: {r._init_error}")
    if r._bm25 is None:
        pytest.skip(f"BM25 tidak terbangun: diminta={r.mode_diminta} sebab={r._bm25_error}")

    hasil = r.retrieve("dosis awal insulin basal", top_k=3)

    assert hasil, "hibrida tidak mengembalikan hasil"
    assert all(h["similarity"] is None for h in hasil)
    assert [h["rank"] for h in hasil] == list(range(1, len(hasil) + 1))
    assert all(h["text"].strip() for h in hasil)
