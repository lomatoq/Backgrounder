# Backgrounder — Master Spec (для Claude Code)

> Версія: 2026-05-31. Мова: BY + EN-тэрміны.
> Мэта: не «яшчэ адзін rembg», а **прадукт з моатам** — разумны раўтэр + decontamination-ядро.
> Гэты файл — гатовы брыф. Адкрый яго ў Claude Code у рэпа `Backgrounder` і ідзі па фазах.

---

## 0. TL;DR — рашаючы фактар

**Перамогу дае не лепшая α-маска, а сумеснае аднаўленне Foreground-колеру (decontamination).**

Усе адкрытыя інструменты (BiRefNet, BEN2, SDMatte) аддаюць толькі `α` і пакідаюць
аднаўленне `F` на пост-апрацоўку — таму ва ўсіх той самы сіні/зялёны rim на краі.
Матэматычны корань усіх праблем (rim, паўпразрыстасць, шкло, тэкст на градыенце):

```
I(x) = α(x)·F(x) + (1 − α(x))·B(x)
```

Калі на выхад ідзе `I` замест `F`, фон «прасочваецца» ў паўпразрыстыя пікселі.
**Хто аддае чысты `F` разам з `α` — выйграе.** Гэта ядро ўсяго праекта.

Другі моат — **раўтынг па рэгіёнах** (spatial mixture-of-experts): у адной выяве
адначасова бываюць сіні rim + тэкст + валасы; адзіны эксперт іх усе не возьме.
Раскладаем выяву на зоны па тыпе failure → кожную зону свайму эксперту → soft-fusion.

Мадэлі ўсе публічныя → перавагу дае **аркестрацыя**, а не вага з HuggingFace.
Гэта тое, што нельга проста сцягнуць.

---

## 1. Архітэктура верхняга ўзроўню

```
                    ┌─────────────────────────────────────────┐
  input image  ───► │ 1. BASE PASS (cheap)                     │
                    │    BiRefNet-HR + BEN2  →  α0, α0'         │
                    └───────────────┬─────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │ 2. FAILURE-MAP ANALYZER (no-reference)    │
                    │    disagreement / transition / TTA-var /  │
                    │    border-flatness+chroma / stroke-HF     │
                    │    →  per-pixel difficulty + region labels │
                    └───────────────┬─────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │ 3. ROUTER  (cost-sensitive, λ = режим)    │
                    │    e*(r)=argmax_e[ΔQ_e(feat_r) − λ·cost_e] │
                    └───────────────┬─────────────────────────┘
                                    │  (per-region dispatch)
        ┌──────────────┬───────────┼────────────┬──────────────┐
        ▼              ▼           ▼            ▼              ▼
   FLAT/CG KEY    FINE/HAIR     TEXT/LOGO    CONCEPT       (base kept)
   unmix-solver   ZIM/SDMatte*  crisp key    SAM 3.1
        └──────────────┴───────────┼────────────┴──────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │ 4. DECONTAMINATION CORE  (F-recovery)     │
                    │    closed-form  →  FBA  →  (DRIP опц.)     │
                    └───────────────┬─────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │ 5. SOFT FUSION  α*(x)=Σ w_e(x)α_e/Σ w_e   │
                    │    + feather праз граніцы рэгіёнаў         │
                    └───────────────┬─────────────────────────┘
                                    ▼
                          RGBA = (F_recovered, α*)
```

---

## 2. Модулі — дэталі і матэматыка

### 2.1 Failure-Map Analyzer (`backgrounder/analyze/failure_map.py`)

No-reference карта цяжкасці з артаганальных сігналаў (кожны лавіць свой клас памылкі):

| Канал | Формула / метад | Лавіць |
|---|---|---|
| `U_dis` disagreement | `|α_BiRefNet − α_BEN2|` | агульная нявызначанасць |
| `U_trans` transition mass | індыкатар `α ∈ (ε, 1−ε)` | мяккія краі (валасы/шкло) |
| `U_tta` TTA-варыяцыя | дысперсія α па flip/scale аўгментацыях базы | нестабільнасць мадэлі |
| `U_flat` border-flatness | low color-variance + connected-to-border + chroma key score | flat/CG rim (сіні фон) |
| `U_text` stroke/HF | Laplacian energy + Stroke-Width-Transform | тэкст, тонкія лініі, лога |

Выхад: стэк нармалізаваных карт `U ∈ [0,1]^{H×W×5}` + дыскрэтныя `region_labels`
(watershed/connected-components па дамінантным канале).

> Гэта навуковае ядро. Кожны канал = асобны юніт-тэст на сінтэтычным кейсе.

### 2.2 Router (`backgrounder/route/policy.py`)

Фармалізацыя — **learning-to-defer / cost-sensitive dispatch** на рэгіён `r`:

```
e*(r) = argmax_e [ ΔQ_e(features_r) − λ · cost_e ]
```

- `ΔQ_e` — чаканы прырост якасці над базай (спачатку эўрыстыка, потым вучаны прэдыктар).
- `cost_e` — латэнцыя/VRAM эксперта.
- `λ` — **гэта літаральна UI-рэжымы**: `Fast` (вялікі λ, мала дарагіх экспертаў) /
  `Smart Auto` (баланс) / `Max Quality` (λ→0).

Эвалюцыя палітыкі:
1. **v1** — рукамі-падабраныя парогі на failure-map (як зараз, але per-region).
2. **v2** — лёгкі `gating-net g(features_r) → expert`, навучаны мінімізаваць
   фінальную памылку кампазіту; эксперты замарожаныя (танны трэйн).
3. **v3** — contextual bandit / value-net, прадказвае `ΔQ_e` напрост;
   reward = рэальны прырост якасці − λ·латэнцыя.

### 2.3 Decontamination Core — **галоўнае** (`backgrounder/decontam/`)

Тры ўзроўні, будуй па чарзе:

**Level 1 — closed-form unmixing (`unmix.py`)** — зараз, без трэйна:
```
F(x) = (I(x) − (1 − α(x))·B̂) / α(x),   для α(x) > τ
```
- `B̂` — clean-plate ацэнка фону (глабальная або per-region з border-пікселяў).
- Дзе `α → 0` раўнанне ill-conditioned → кламп + прапагацыя суседняга `F` (band-by-band,
  ад foreground-мяжы вонкі) + edge-shrink. Гэта класічны color-decontamination.
- Despill fallback: душы screen-канал там, дзе ён перавышае іншыя на краі.
- **Замяняе CorridorKey** для камерцыі (той CC BY-NC-SA + green-first; нам не падыходзіць).

**Level 2 — FBA Matting (`fba.py`)** — вучаны апгрэйд:
- Сетка прадказвае `F, B, α` адначасова (arXiv 2003.07711, адкрыты код/вагі).
- Дае правільны `F` у складаных краях, дзе closed-form шумее.
- Інтэграцыя: бярэ `I` + trimap (з α0 праз адаптыўны erosion/dilation) → `(F, B, α)`.

**Level 3 — diffusion prior (опц., bleeding-edge) (`drip.py`)**:
- **DRIP** (NeurIPS 2024): LDM з cross-domain switcher, за адзін праход `F` + `α`,
  latent-transparency decoder аднаўляе HF пасля VAE-кампрэсіі.
- **LayerDecomp** (Adobe, CVPR 2025): генератыўная дэкампазіцыя ў слаі,
  захоўвае цені/адлюстраванні ў FG-слаі. Для прэмаўм «Max Quality» рэжыму.
- Цяжкае; гейтаваць толькі на самыя складаныя кейсы.

### 2.4 Adaptive Trimap (`backgrounder/decontam/trimap.py`)

Крытычны крок, які большасць пропускае (і праз які валіцца cascade):
- erode(α0) → definite FG; dilate(α0) → unknown band.
- **Шырыня band — адаптыўная**: шырэй дзе `U_trans`/`U_text` высокія (валасы),
  вузей на гладкіх краях. Не фіксаваны kernel!

### 2.5 Soft Fusion (`backgrounder/fuse/blend.py`)

Без швоў на стыку рэгіёнаў:
```
α*(x) = Σ_e w_e(x)·α_e(x) / Σ_e w_e(x)
```
- `w_e(x)` — упэўненасць эксперта (з uncertainty або softmax gating-net).
- Feather вагі праз граніцы рэгіёнаў (Gaussian на label-boundary).

---

## 3. Роўстэр экспертаў (2026)

| Эксперт | Роля | Ліцэнзія / нататка |
|---|---|---|
| **BiRefNet-HR** | база, чыстая маска агульных аб'ектаў | open; ужо ў рэпа |
| **BEN2** | second opinion → disagreement signal | open; ужо ў рэпа |
| **FBA Matting** | decontamination-эксперт (F,B,α) | open (2003.07711) — **ядро** |
| **unmix-solver** | flat/CG rim, closed-form F | свой код, 0 ліцэнзіі |
| **ZIM** | fine/micro матэ, тонкія структуры, зеро-шот | open, NAVER, ICCV 2025 (наш `2411.00626`) |
| **SDMatte\*** | цяжкі fine-detail matting (gated) | MIT; чэкпойнт COCO-Matte, не Composition |
| **SAM 3.1** | концепт/тэкст/бокс-выбар (advanced) | open, Meta; патрабуе server-GPU |
| **crisp keyer** | тэкст/лога → рэзкая 0/1 альфа + connectivity | свой код |

**Тэкст-нюанс:** для тэксту/лога патрэбна **рэзкая альфа (0/1), НЕ feathered** —
soft-matting размывае штрыхі (раўнанне ill-posed на тонкіх структурах).
Раўтэр мусіць класіфікаваць crisp-vs-soft па `U_text`.

---

## 4. Eval-набор — твой сапраўдны актыў (`eval/`)

Без гэтага раўтэр не стане разумным і gate не навучыцца.
- Збяры **200–500 сваіх цяжкіх кейсаў** з ground-truth α (і па магчымасці F):
  сіні/зялёны rim, тэкст на фоне, валасы, шкло/празрыстасць, CG/скрыншоты.
- Метрыкі: SAD, MSE, Grad, Conn (стандарт) **плюс** Foreground-color error у band
  (бо decontamination — наша фішка, мерай менавіта яе).
- Знешнія бенчмаркі для параўнання з паперамі: AIM-500, AM-2K, P3M-500,
  MicroMat-3K (ZIM), RefMatte-RW100.
- **Гэты набор кампаундзіцца**: мадэлі прыходзяць/сыходзяць, бенчмарк + палітыка застаюцца.

Дадатковыя трэйн-датасеты (калі дойдзе да навучання gate/FBA-файнцюна):
Distinctions-646, Composition-1k (па ліцэнзіі — пісаць bprice@adobe.com),
COCO-Matting (SEMat), AM-2K, P3M-10k, Transparent-460, BG-20K (фоны).

---

## 5. План па фазах (для Claude Code)

### Фаза 0 — Decontamination Core L1 (НАЙБОЛЬШЫ хуткі выйгрыш)
- [ ] `decontam/unmix.py` — closed-form F-recovery + clean-plate `B̂` + edge-reg.
- [ ] `decontam/trimap.py` — adaptive band.
- [ ] Уторкнуць у існуючы flat/CG маршрут замест чыстага chroma-cleanup.
- [ ] Юніт-тэсты: blue-halo removal, dark-outline preserve, enclosed-blue preserve.
- **Вынік:** сіні rim знікае на большасці кейсаў. Без трэйна.

### Фаза 1 — Failure-Map + Per-Region Router
- [ ] `analyze/failure_map.py` — 5 каналаў + region_labels.
- [ ] `route/policy.py` v1 (эўрыстыкі) + інтэграцыя λ ↔ UI-рэжымы.
- [ ] `fuse/blend.py` — soft fusion + feather.
- **Вынік:** адна выява = некалькі экспертаў паралельна, без швоў.

### Фаза 2 — FBA decontamination (L2) + ZIM эксперт
- [ ] `decontam/fba.py` — інтэграцыя FBA (F,B,α), вагі.
- [ ] `experts/zim.py` — fine/micro матэ і тэкст.
- [ ] Crisp-keyer для тэксту.
- **Вынік:** правільны F у складаных краях; тэкст рэзкі.

### Фаза 3 — Eval-набор + вучаны gate (v2)
- [ ] `eval/` харнэс + 200–500 курыраваных кейсаў з GT.
- [ ] `route/policy.py` v2 — gating-net, навучаны на eval-loss.
- **Вынік:** раўтэр перастае быць эўрыстыкай, пачынае вымярацца і паляпшацца.

### Фаза 4 (опц.) — diffusion prior для «Max Quality»
- [ ] `decontam/drip.py` (DRIP) / LayerDecomp інтэграцыя, gated.
- [ ] (Опц.) SAMA-як-эксперт: frozen-SAM + MVLE + Local-Adapter + dual-head,
      калі патрэбен адзін forward = seg+matte. Будаваць форкам SEMat, не з нуля.

---

## 6. UI (захаваць простым)
- `Smart Auto` (дэфолт) / `Fast` / `Max Quality` = тры значэнні λ.
- Developer-кантролі пад accordion (failure-map оверлеі, выбар эксперта).

---

## 7. Прынцыпы (не парушаць)
1. **Аддавай F, не I.** Кожны выхадны піксель з `α<1` мусіць быць decontaminated.
2. **Раўтынг па рэгіёнах, не па выяве.**
3. **Adaptive trimap, не фіксаваны kernel.**
4. **Soft fusion, не жорсткае пераключэнне** (швы).
5. **Crisp для тэксту, soft для валасоў.**
6. **Eval-набор — раней за оптымізацыю.** Нельга паляпшаць тое, што не мераеш.
7. **Без CorridorKey у камерцыі** (ліцэнзія). Свой unmix-solver.

---

## 8. Ключавыя крыніцы
- FBA Matting — arXiv 2003.07711 (joint F,B,α; open)
- DRIP — NeurIPS 2024 (diffusion prior, F+α адзін праход)
- LayerDecomp — CVPR 2025, Adobe (layer decomposition, цені/адлюстраванні)
- DiffDecompose / AlphaBlend — arXiv 2505.21541 (semi-transparent decomposition)
- ZIM — arXiv 2411.00626, ICCV 2025, NAVER (zero-shot micro matte; open)
- SDMatte — arXiv 2508.00443, ICCV 2025 (diffusion interactive matting; MIT)
- SEMat — arXiv 2410.06593, TCSVT 2026 (real-scenario prior; open + COCO-Matting)
- SAMA — AAAI 2026 (unified seg+matte, dual-head; код не выпушчаны)
- SAM 3 / 3.1 — github.com/facebookresearch/sam3 (concept segmentation; open)
- BiRefNet — AIR 2024; BEN2; BRIA RMBG-2.0
- Color decontamination — US Patent 8379972 (band-by-band F estimation)

---

## 9. Першы крок у Claude Code
```
Прачытай BACKGROUNDER_MASTER_SPEC.md.
Пакажы бягучую структуру backgrounder/ і дзе зараз робіцца flat-background cleanup.
Потым рэалізуй Фазу 0: decontam/unmix.py + decontam/trimap.py,
уторкні ў flat/CG маршрут, дадай юніт-тэсты з §5 Фаза 0.
```
