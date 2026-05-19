
---

## Cara Menjalankan Bot

### Prasyarat (sekali saja)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Salin env template
copy .env.example .env
# Edit .env dengan nilai-nilaimu
```

---

### 🟡 Paper Trading (Mode Default)

Tidak butuh API Polymarket. Semua bet disimulasikan secara lokal.

**Step 1 — Pastikan `.env` berisi:**
```env
PAPER_MODE=true
STARTING_CAPITAL=10.00
```

**Step 2 — Jalankan infrastruktur (DB + Redis):**
```bash
docker-compose up -d timescaledb redis
```

**Step 3 — Jalankan bot:**
```bash
python -m src.main
```

**Step 4 (opsional) — Dashboard:**
```bash
# Terminal kedua
uvicorn src.dashboard.api:app --port 8000 --reload

# Terminal ketiga (frontend)
cd frontend && npm install && npm run dev
# → buka http://localhost:3000
```

---

### 🔴 Live Trading

**Validasi wajib sebelum live:**
- ≥50 bets paper trading
- Win rate >53%

**Step 1 — Daftar dan dapatkan API key:**
- [Polymarket API Key](https://docs.polymarket.com/) — untuk eksekusi bet
- Telegram Bot token — untuk alert real-time

**Step 2 — Update `.env`:**
```env
PAPER_MODE=false
STARTING_CAPITAL=10.00
POLYMARKET_API_KEY=your_key_here
POLYMARKET_PRIVATE_KEY=your_private_key_here
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Step 3 — Jalankan infrastruktur:**
```bash
docker-compose up -d timescaledb redis
```

**Step 4 — Jalankan bot:**
```bash
python -m src.main
```

Bot akan menampilkan warning 3 detik sebelum mulai live trading.

---

### Perintah Berguna

```bash
# Cek semua service berjalan
docker-compose ps

# Lihat log database
docker-compose logs -f timescaledb

# Stop semua service
docker-compose down

# Jalankan semua tests (verify sebelum live)
python -m pytest tests/ --ignore=tests/perf -q
```

---

**Catatan penting**: `src/main.py` saat ini menginisialisasi semua komponen (RingBuffer, StateManager, AlertManager, DB pool) namun `ExecutionEngine` penuh membutuhkan feature engine dan model yang sudah di-train. Untuk production penuh, wire `_initialize_components()` dengan model yang sudah tersedia di `models/latest/`.