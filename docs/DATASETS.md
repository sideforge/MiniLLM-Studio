# Datensatz-Katalog

Alle 41 Datensätze wurden gegen den Hugging Face Hub getestet (Streaming, die ersten Zeilen lesbar).
Die App lädt immer nur so viele Dokumente, wie du einstellst – nichts wird komplett heruntergeladen.
Zusätzlich kannst du in der App **beliebige eigene Datensätze** vom Hub hinzufügen (siehe README).

| Name | Kategorie | Sprache | Grösse | Schwierigkeit | Hugging-Face-Repo |
|---|---|---|---|---|---|
| **Alpaca (Anweisungen)** | Anweisungen | EN | 52k Beispiele | mittel | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) |
| **Dolly 15k (Anweisungen)** | Anweisungen | EN | 15k Beispiele | mittel | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| **Guanaco (mehrsprachiger Chat, 1k)** | Anweisungen | mehrsprachig | 1k Beispiele | mittel | [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) |
| **OpenAssistant (mehrsprachig)** | Anweisungen | mehrsprachig | ~85k Nachrichten | schwer | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) |
| **SlimOrca (GPT-4-Antworten)** | Anweisungen | EN | ~518k Dialoge | schwer | [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| **SmolTalk (Chat-Mix von Hugging Face)** | Anweisungen | EN | ~1 Mio. Dialoge | sehr schwer | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) |
| **Tulu-3 SFT-Mischung** | Anweisungen | mehrsprachig | ~940k Dialoge | sehr schwer | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| **CodeAlpaca (20k)** | Code | EN | 20k Beispiele | leicht | [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
| **CodeFeedback (Code-Fragen, viele Sprachen)** | Code | EN | ~156k Beispiele | schwer | [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) |
| **Evol-Instruct Code (80k)** | Code | EN | 80k Beispiele | schwer | [nickrosh/Evol-Instruct-Code-80k-v1](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) |
| **Magicoder (Aufgabe + Lösung, viele Sprachen)** | Code | EN | 75k Beispiele | mittel | [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) |
| **Python-Code (18k)** | Code | EN | 18k Beispiele | schwer | [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) |
| **Python-Codes (25k)** | Code | EN | 25k Beispiele | leicht | [flytech/python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k) |
| **Python-Quellcode von GitHub (CodeParrot)** | Code | EN | ~5 Mio. Dateien | mittel | [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) |
| **Self-OSS-Instruct (Python, geprüft)** | Code | EN | 50k Beispiele | mittel | [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) |
| **Cosmopedia Geschichten** | Geschichten | EN | ~600k Geschichten | mittel | [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) |
| **Project Gutenberg (Bücher)** | Geschichten | EN | ~50k Bücher | schwer | [manu/project_gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) |
| **Tiny Shakespeare** | Geschichten | EN | ~1 MB | mittel | [Trelis/tiny-shakespeare](https://huggingface.co/datasets/Trelis/tiny-shakespeare) |
| **TinyStories** | Geschichten | EN | 2.1 Mio. Geschichten | sehr leicht | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| **AG News** | Nachrichten | EN | 120k Meldungen | mittel | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) |
| **CNN / Daily Mail** | Nachrichten | EN | ~300k Artikel | schwer | [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) |
| **Tagesschau-Artikel (Deutsch)** | Nachrichten | DE | ~20k Artikel | mittel | [bjoernp/tagesschau-2018-2023](https://huggingface.co/datasets/bjoernp/tagesschau-2018-2023) |
| **Englische Zitate** | Sonstiges | EN | ~2.5k Zitate | leicht | [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) |
| **IMDB Filmkritiken** | Sonstiges | EN | ~65 MB | mittel | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) |
| **Rotten Tomatoes Sätze** | Sonstiges | EN | ~1 MB | mittel | [cornell-movie-review-data/rotten_tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) |
| **C4 (Common Crawl, bereinigt)** | Web | EN | 365 Mio. Dokumente | schwer | [allenai/c4](https://huggingface.co/datasets/allenai/c4) |
| **FineWeb-2 Deutsch** | Web | DE | Milliarden Tokens | schwer | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| **FineWeb-Edu** | Web | EN | 10 Mrd. Tokens (Sample) | schwer | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| **OpenWebText (GPT-2-Daten)** | Web | EN | 8 Mio. Dokumente | schwer | [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext) |
| **The Pile (ohne Urheberrechtsmaterial)** | Web | EN | ~1 Bio. Zeichen | schwer | [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted) |
| **Cosmopedia (100k)** | Wissen | EN | 100k Texte | mittel | [HuggingFaceTB/cosmopedia-100k](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k) |
| **Cosmopedia v2 (SmolLM-Korpus)** | Wissen | EN | 39 Mio. Texte | mittel | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| **FineMath (Mathematik-Webseiten)** | Wissen | EN | ~10 Mrd. Tokens | sehr schwer | [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) |
| **GSM8K (Mathe)** | Wissen | EN | 7.5k Aufgaben | schwer | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| **OpenMathInstruct-2** | Wissen | EN | ~14 Mio. Beispiele | sehr schwer | [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |
| **OpenR1-Math (Aufgaben + Lösungsweg)** | Wissen | EN | 220k Aufgaben | sehr schwer | [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) |
| **OpenWebMath** | Wissen | EN | 6 Mio. Dokumente | schwer | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) |
| **WikiText-103** | Wissen | EN | ~540 MB | mittel | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) |
| **WikiText-2** | Wissen | EN | ~11 MB | mittel | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) |
| **Wikipedia (Deutsch)** | Wissen | DE | 2.8 Mio. Artikel | schwer | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) |
| **Wikipedia (Englisch)** | Wissen | EN | 6.4 Mio. Artikel | schwer | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) |

## Beschreibungen

- **Alpaca (Anweisungen)** – 52k Frage-Antwort-Paare im Anweisungs-Format (von Stanford).
- **Dolly 15k (Anweisungen)** – 15k von Menschen geschriebene Anweisungen und Antworten.
- **Guanaco (mehrsprachiger Chat, 1k)** – 1000 Chat-Beispiele in mehreren Sprachen – winzig.
- **OpenAssistant (mehrsprachig)** – Von Freiwilligen geschriebene Chat-Nachrichten in vielen Sprachen (auch Deutsch).
- **SlimOrca (GPT-4-Antworten)** – ~500k Aufgaben mit ausführlichen Antworten von GPT-4, mit Begründung Schritt für Schritt.
- **SmolTalk (Chat-Mix von Hugging Face)** – Grosse Mischung aus Dialogen, Aufgaben, Code und Mathe – damit werden SmolLM-Chatmodelle feingetrimmt.
- **Tulu-3 SFT-Mischung** – Die Anweisungs-Mischung des Allen Institute (Wissen, Code, Mathe, Sicherheit).
- **CodeAlpaca (20k)** – 20k kurze Code-Aufgaben – klein, sauber, ideal zum schnellen Ausprobieren.
- **CodeFeedback (Code-Fragen, viele Sprachen)** – Gefilterte Programmier-Fragen mit Antworten – Python, Java, C++, JavaScript, SQL u. a.
- **Evol-Instruct Code (80k)** – 80k immer schwieriger gemachte Programmieraufgaben mit ausführlichen Antworten.
- **Magicoder (Aufgabe + Lösung, viele Sprachen)** – 75k Programmieraufgaben mit Lösungen in Python, C++, Java, Rust, TypeScript u. a. – von einem grossen Modell aus echtem Open-Source-Code erzeugt.
- **Python-Code (18k)** – Python-Aufgaben mit Lösungs-Code.
- **Python-Codes (25k)** – 25k kurze Python-Skripte mit Aufgabenbeschreibung.
- **Python-Quellcode von GitHub (CodeParrot)** – Echte, bereinigte Python-Dateien aus GitHub-Projekten. Die beste Wahl, um ein Modell Code schreiben zu lassen (Dateien werden auf 6000 Zeichen gekürzt).
- **Self-OSS-Instruct (Python, geprüft)** – 50k Python-Aufgaben, deren Lösungen automatisch ausgeführt und getestet wurden.
- **Cosmopedia Geschichten** – Längere, vielfältigere Geschichten als TinyStories (von Mixtral geschrieben).
- **Project Gutenberg (Bücher)** – Gemeinfreie Romane und Sachbücher. Lange, zusammenhängende Literatur-Sprache (jedes Buch wird auf 30'000 Zeichen gekürzt).
- **Tiny Shakespeare** – Theaterstücke von Shakespeare (ca. 1 MB). Sehr klein, ideal für schnelle Tests.
- **TinyStories** – Kurze, einfache Kindergeschichten (von GPT-4 geschrieben). Der Klassiker, um zu zeigen, dass schon winzige Modelle ordentliche Texte lernen.
- **AG News** – Kurze Nachrichtenmeldungen aus vier Themenbereichen.
- **CNN / Daily Mail** – Lange Nachrichtenartikel mit Zusammenfassung.
- **Tagesschau-Artikel (Deutsch)** – Nachrichtenartikel der Tagesschau von 2018 bis 2023 – gut für deutschen Zeitungsstil.
- **Englische Zitate** – Berühmte Zitate mit Autor – winzig und witzig.
- **IMDB Filmkritiken** – 50'000 Filmkritiken. Informelle Sprache mit Meinungen.
- **Rotten Tomatoes Sätze** – Kurze Filmkritik-Sätze – sehr klein.
- **C4 (Common Crawl, bereinigt)** – Riesiger bereinigter Web-Text, mit dem auch T5 trainiert wurde.
- **FineWeb-2 Deutsch** – Gefilterter deutscher Web-Text (Nachrichten, Blogs, Foren, Shops). Das beste Material für ein deutsches Modell.
- **FineWeb-Edu** – Gefilterte, bildungsrelevante Webseiten – wird für moderne Open-Source-Modelle benutzt.
- **OpenWebText (GPT-2-Daten)** – Nachbau der Trainingsdaten von GPT-2: Webseiten, die auf Reddit verlinkt wurden.
- **The Pile (ohne Urheberrechtsmaterial)** – Bunter Mix aus Web, Büchern, arXiv-Papern, GitHub-Code, Foren und Wikipedia – wie ein Mini-Internet.
- **Cosmopedia (100k)** – Synthetische Lehrbuch-Texte, Blogposts und Geschichten, von einem grossen Modell geschrieben. Sehr saubere Daten.
- **Cosmopedia v2 (SmolLM-Korpus)** – Hochwertige synthetische Lehrbuch-Texte, Blogposts und Erklärungen – damit wurde SmolLM trainiert. Sehr gut, um ein kleines Modell zusammenhängend schreiben zu lassen.
- **FineMath (Mathematik-Webseiten)** – Sorgfältig gefilterte Webseiten mit Mathematik, Formeln und Herleitungen – Trainingsdaten moderner Open-Source-Modelle.
- **GSM8K (Mathe)** – Mathe-Textaufgaben mit Lösungsweg. Für kleine Modelle fast unmöglich – aber lehrreich!
- **OpenMathInstruct-2** – Millionen Mathe-Aufgaben mit Lösungen (NVIDIA).
- **OpenR1-Math (Aufgaben + Lösungsweg)** – 220k anspruchsvolle Mathe-Aufgaben (Wettbewerbs-Niveau) mit ausführlicher Lösung.
- **OpenWebMath** – Mathematik-Webseiten mit Formeln (LaTeX), Beweisen und Erklärungen.
- **WikiText-103** – Die grosse Variante von WikiText: 100 Mio. Wörter geprüfter Wikipedia-Artikel.
- **WikiText-2** – Saubere englische Wikipedia-Artikel (Qualitäts-Artikel). Kleiner Standard-Benchmark der Sprachmodell-Forschung.
- **Wikipedia (Deutsch)** – Deutsche Wikipedia-Artikel (Stand November 2023). Schwierig für kleine Modelle, aber spannend für deutsche Texte.
- **Wikipedia (Englisch)** – Englische Wikipedia-Artikel (Stand November 2023).
