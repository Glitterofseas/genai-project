<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Logo" width="110" height="110">
</p>

<h1 align="center">SMS Recruiting Chatbot</h1>

<p align="center">
  A multi-agent GenAI bot that screens Python Developer candidates over SMS and books their interview.<br>
  <b>LangChain · OpenAI · ChromaDB · SQL · Streamlit</b>
</p>

---

## Table of Contents

- [Project Purpose](#project-purpose)
- [How It Works](#how-it-works)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data and Resources](#data-and-resources)
- [Design Notes](#design-notes)
- [Known Deviations](#known-deviations)
- [License](#license)

---

## Project Purpose

An SMS-based chatbot that interacts with job candidates for a Python Developer
position. It gathers and verifies information, answers questions about the role,
and ultimately either **schedules an interview with a human recruiter** or
**politely ends the conversation**.

A **Main Agent** manages the dialogue turn by turn and chooses one of three
actions - `continue`, `schedule`, `end` - by consulting three specialised
**Advisor agents**.

---

## How It Works

One turn of the conversation, following `docs/workflow/`:

```
        Start
          |
   +------+------+
   |             |
 User        Fill Registration
 responds        Form
   |             |
   +------+------+
          v
   Main Agent: Receives and Processes Input
          |
          v
   Decides 1 of 3  ------------------------------+
          |                                      |
   +------+-------+---------------+              |
   v              v               v              |
 Exit Advisor  Sched Advisor   Info Advisor      |
   |              |               |              |
 End? / Not    Sched? / Not    Needed? / Not     |
   |              |  \            |  \           |
   |              |   -> SQL      |   -> Chroma  |
   +------+-------+---------------+              |
          v                                      |
   Main Agent: Receives and Processes Input      |
          |                                      |
          v                                      |
   Decides 1 of 2 ---- consult advisor again ----+
          |
     send output
          v
    Sends Output to User -> End of turn
```

Two properties of this design are load-bearing:

- **One advisor per iteration.** The Main Agent routes to exactly one advisor,
  reads its verdict, then decides whether to consult another or reply. The loop
  is capped at three advisor calls per turn.
- **Retrieval is conditional.** The SQL schedule is queried only after the
  Scheduling Advisor decides to schedule; Chroma is queried only after the Info
  Advisor decides information is needed. A turn that needs neither costs nothing
  beyond the classification calls.
- **The schedule is reached by function calling.** Once the Scheduling Advisor
  votes to schedule, a LangChain agent with two bound tools takes over: the
  model decides which to call and with what arguments.

| Advisor | Binary decision | Tool it may use |
|---|---|---|
| **Conversation Exit** | End / Don't End | - |
| **Interview Scheduling** | Sched / Don't Sched | SQL schedule, via LangChain tools the model calls |
| **Conversation Info** | Info Needed / Not Needed | Chroma vector store |

---

## Results

Evaluated on all 59 labelled recruiter turns, teacher-forced and read-only.
Full analysis in [`tests/test_evals.ipynb`](tests/test_evals.ipynb).

| System | All 59 turns | Held-out (conv 11-15) |
|---|---|---|
| Baseline - always `continue` | 42.4% | - |
| Baseline - **last turn = `end`** | 67.8% | - |
| Rule-based agent (no API calls) | 79.7% | 89.5% |
| **LLM multi-agent (LangChain + OpenAI)** | **85.2%** | **89.5%** |

The LLM figures are the **mean of four runs** (84.7%-86.4% overall,
84.2%-94.7% held out). `temperature=0` does not make OpenAI
deterministic, so a single run is not a reliable figure. Only **3 of the 59
turns** ever changed prediction across all four runs - the pipeline is stable -
but on the 19-turn held-out split a single turn is 5.3 points, which is why the
held-out range is so much wider than the overall one.

Note that the best run (86.4% / 94.7%) is the one with SQL function calling, and
that is **not** evidence the tools improved accuracy: the action is decided by
the binary verdict in phase 1, before any tool is called, so the tools cannot
influence it. Its one extra correct turn is drift, not a gain.

Read these numbers with two caveats, both documented in the notebook:

1. **`end` is easier than it looks.** Every conversation's final recruiter turn
   is labelled `end`, so a one-line positional rule already scores 100% recall
   on that class. That is why the trivial baselines are reported alongside.
2. **The held-out split is small.** 19 turns - one turn is worth 5.3 accuracy
   points, so held-out differences of two or three turns are noise.

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key (chat + embeddings)
- *Optional:* SQL Server, only to run the provided `db_Tech.sql` yourself

### Installation

```bash
git clone <your-repo-url>
cd Project
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

Create `.env` from the template and add your key:

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=sk-proj-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
SCHEDULE_BACKEND=sqlite
```

### One-time setup

```bash
python -m app.modules.db.build_fixture     # build the schedule (already committed)
python -m app.modules.embedding.build_index   # embed the PDF into Chroma
```

The embedding step is the only place embeddings are paid for; the chat loop
never re-embeds the corpus.

---

## Usage

**Streamlit (the proof of concept):**

```bash
streamlit run streamlit_app/streamlit_main.py
```

Fill the registration form - or skip it - then chat as the candidate. The
sidebar switches between the LLM and the free rule-based agent, and the advisor
trace under each reply shows which advisors were consulted and which slots came
back from SQL.

**Terminal:**

```bash
python -m app.main --date 2024-04-30
```

**Evaluation:**

```bash
python -m app.modules.evaluation.run_eval --system rule   # free
python -m app.modules.evaluation.run_eval --system llm    # spends tokens
```

**Tests:**

```bash
python -m pytest -q     # all offline, no API key needed
```

---

## Project Structure

```text
Project/
├── .env / .env.example        # secrets (gitignored) and its template
├── requirements.txt
├── data/
│   ├── sms_conversations.json      # labelled evaluation set
│   ├── Python Developer Job Description.pdf
│   ├── db_Tech.sql                 # provided T-SQL schedule script
│   ├── schedule.sqlite             # frozen seed; the reproducibility anchor
│   ├── schedule.local.sqlite       # app's writable copy (gitignored)
│   └── fine_tuning/*.jsonl         # Exit Advisor training data
├── docs/
│   └── workflow/                   # the workflow diagram, 4 sections
├── app/
│   ├── main.py                     # CLI entry point
│   └── modules/
│       ├── config/settings.py      # env, paths, the fixed train/test split
│       ├── db/                     # rules, dates, store, fixture builder
│       ├── embedding/              # offline PDF -> Chroma, retriever
│       ├── agents/                 # main agent, advisors, prompts, tools, memory
│       ├── fine_tuning/            # JSONL builder, job runner
│       └── evaluation/             # dataset, harness, run_eval
├── streamlit_app/
│   ├── streamlit_main.py           # registration form + chat UI
│   └── utils.py
├── tests/
│   ├── test_main.py                # tests for the main application
│   └── test_evals.ipynb            # Accuracy + Confusion Matrix
└── results/*.json                  # cached evaluation runs the notebook reads
```

---

## Data and Resources

The supplied materials are in the repository byte-identical to what was
provided; only their locations changed to fit the required structure. The
workflow diagram is in `docs/workflow/`.

| Resource | Role |
|---|---|
| `sms_conversations.json` | 15 conversations, 103 turns; 59 labelled recruiter turns |
| `Python Developer Job Description.pdf` | Embedded into Chroma; the Info Advisor's knowledge |
| `db_Tech.sql` | Builds `Tech.dbo.Schedule` - 2024, Tue-Fri + Sun, 09:00-17:00, 4 positions |
| Chroma | Local persistent instance in `chroma_db/` |
| LangChain | Agents, memory, tools |
| OpenAI | Chat and embeddings |

---

## Design Notes

**Dates are anchored to the conversation, never the wall clock.**
`db_Tech.sql` populates 2024 and the transcripts are from April 2024. Resolving
"next Friday" against today's date would return zero rows every time, so every
resolution hangs off the turn's own timestamp. See `app/modules/db/dates.py`.

**Multi-slot offers are parsed positionally.** Recruiter turns routinely offer
two slots - *"this Friday at 11 AM or next Monday at 9 AM"* - so each date is
paired with the time that follows it and precedes the next date. Pairing them
independently produces a phantom "Monday at 11 AM".

**Mondays and Saturdays do not exist.** `db_Tech.sql` excludes them, yet the
transcripts contain five offers of *"next Monday at 9 AM"* and three candidate
acceptances of *"Monday at 3 PM"*. `check_slot()` returns `None` rather than
"unavailable" for these, so the bot counter-offers instead of confirming an
impossible booking. The bot therefore behaves differently from the transcripts
here - correctly.

**Scheduling is proactive.** 10 of the 19 `schedule` turns follow the
candidate's first answer with no request from them at all. A reactive advisor
that waits for the candidate to raise timing scores 0.05 recall on that class.

**The schedule is frozen.** `db_Tech.sql` seeds availability with
`ABS(CHECKSUM(NEWID()))`, so every run of the script produces a *different*
calendar and no reported number would survive a re-run. The table is therefore
built once and committed as `data/schedule.sqlite`; 2026-2027 rows are generated
under the same rules so live demos have bookable dates.

**A confirmation means a real, verified booking.** When a candidate accepts,
the named time is resolved - their own words win, and a bare "11 AM works" is
matched against the slot it refers to - then checked against the calendar and
written. If the time does not exist or has just been taken, the bot says so and
offers alternatives instead of confirming something that never happened. Without
this the bot announced confirmations while `available` stayed 1 for everyone.

**The app books against a working copy.** `data/schedule.sqlite` is committed
and is what the evaluation reads, so a demo booking must not touch it. The app
writes to a gitignored `schedule.local.sqlite`, seeded from it on first run.

**Evaluation cannot mutate the calendar.** The harness opens the store
read-only, so replaying conversations records booking *intent* without writing.
Otherwise the second evaluation run would score differently from the first.

**Function calling, in two phases.** The Scheduling Advisor keeps the binary
verdict the workflow diagram requires, and only a positive verdict opens the
"SQL Retrieve Sched Options" path. That path is a LangChain agent
(`create_agent`) with two read-only tools over the calendar:

| Tool | Purpose |
|---|---|
| `check_interview_slot(date, time)` | Does this exact slot exist, and is it free? |
| `find_nearest_interview_slots(around_date, around_time, count)` | Up to five real bookable slots near a date |

Two rules make this safe. **The tools are read-only** - a model that could book
would book interviews nobody agreed to, so writes stay deterministic in
`SlotBooker`. And **the slots the candidate is offered come from tool execution,
never from the model's prose**: every row a tool returns is recorded on the
toolkit, so a paraphrased time in the agent's text output can never become an
invitation. If tool calling fails, the advisor falls back to a deterministic
query rather than degrading.

A worked example, from a candidate who proposed an impossible day:

```
check_interview_slot(date='2024-05-06', time='15:00')
  -> Monday 06 May 2024 at 15:00 is NOT on the calendar at all.
find_nearest_interview_slots(around_date='2024-04-30', count=3)
  -> Tuesday 30 Apr 2024 at 12:00 PM; 2:00 PM; 3:00 PM
```

**Conversation memory.** The live CLI and Streamlit app hold the dialogue in
`ConversationMemory`, backed by LangChain's `InMemoryChatMessageHistory`, and
build the advisors' context from it. Evaluation deliberately does not use it -
the harness is teacher-forced from the recorded transcripts, so memory
accumulating across replays would silently change the scores. A test asserts the
harness never imports it.

**Prompting strategies** (`app/modules/agents/prompts.py`): roles in every
system prompt, numbered instruction rules for reproducible verdicts, few-shot
examples drawn only from the training split, and explicit API parameters -
`temperature=0` for classifiers, `0.4` for the candidate-facing composer.

---

## Known Deviations

**Fine-tuning could not be run.** OpenAI closed self-serve fine-tuning to new
jobs; launching one returns:

```
403 training_not_available - "OpenAI is winding down the fine-tuning platform and
your organization is no longer able to create new fine-tuning training jobs."
```

This is a provider-side shutdown, not a billing or code issue - the training
files upload successfully; only job creation is refused. The full pipeline is
implemented in `app/modules/fine_tuning/`: dataset builder (30 training / 14
validation examples), file upload, job launch, polling, and model-id plumbing.
`LLMExitAdvisor` accepts a fine-tuned model id through the interface it already
uses, so a trained model can be dropped in with one flag:

```bash
python -m app.modules.evaluation.run_eval --system llm --exit-model ft:...
```

The Exit Advisor currently runs few-shot, and reaches 1.00 recall on `end`.

**Trained on all `end` labels, not only disengagement.** The spec describes the
Exit Advisor as protecting uninterested candidates, but only 4 of 15
conversations end that way - the other 11 end after a slot is accepted.
Restricting training to disengagement leaves **2** positive examples in the
training split, below OpenAI's own minimum. The advisor therefore owns "should
this end?" in both senses, matching both the label distribution and the diagram.

**`streamlit_app/` rather than `streamlit/`.** A package directory named
`streamlit` shadows the installed `streamlit` package whenever the project root
is on `sys.path` - which breaks `python -m streamlit run ...` and `pytest` from
the root. Verified, then renamed.

**No Streamlit Community Cloud deployment.** The app runs locally.

---

## License

MIT - see [LICENSE](LICENSE).
