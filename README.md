# SMS Recruiting Chatbot - GenAI Project

An SMS chatbot that talks to candidates applying for a Python Developer job.
It answers their questions and either books them an interview or ends the
conversation politely.

Built with LangChain, OpenAI, ChromaDB, SQL and Streamlit.

## Project purpose

A **Main Agent** runs the conversation. On every turn it picks one of three
actions - `continue`, `schedule` or `end` - and asks three **advisor agents**
for help before deciding:

| Advisor | Decides | Uses |
|---|---|---|
| Exit Advisor | should we end the conversation? | - |
| Scheduling Advisor | should we offer an interview time? | the SQL schedule |
| Info Advisor | does the candidate need info about the job? | the PDF in ChromaDB |

The Main Agent asks one advisor at a time, and only asks another if it still
needs to. This follows the workflow diagram in `docs/workflow/`.

The tools are only used when they are needed. SQL is only queried if the
Scheduling Advisor says yes, and ChromaDB only if the Info Advisor says yes.

## Results

Tested on all 59 labelled recruiter turns in `sms_conversations.json`. The full
analysis with the confusion matrices is in `tests/test_evals.ipynb`.

| System | All 59 turns | Held-out (conv 11-15) |
|---|---|---|
| Simple baseline (always `continue`) | 42.4% | - |
| Simple baseline (last turn = `end`) | 67.8% | - |
| Rule-based agent (no API calls) | 79.7% | 89.5% |
| **LLM agent (LangChain + OpenAI)** | **85.2%** | **89.5%** |

The LLM number is the average of four runs, because the result moves a little
between runs even with `temperature=0`.

Two things to keep in mind when reading these:

- The last turn of every conversation is `end`, so a very simple rule already
  gets that class right every time. That is why the simple baselines are in the
  table too.
- The held-out set is only 19 turns, so one turn is worth about 5%.

## How to install and run

You need Python 3.10+ and an OpenAI API key.

```bash
git clone <your-repo-url>
cd genai-project
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your key in it:

```ini
OPENAI_API_KEY=sk-proj-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
SCHEDULE_BACKEND=sqlite
```

Then build the ChromaDB index once. The schedule is already included, so that
step is not needed.

```bash
python -m app.modules.embedding.build_index
```

## Usage examples

Run the Streamlit app:

```bash
streamlit run streamlit_app/streamlit_main.py
```

Fill in the registration form (or skip it) and chat as the candidate. You can
switch between the LLM agent and the free rule-based one in the sidebar, and
open the trace under each reply to see which advisors were asked.

Chat in the terminal instead:

```bash
python -m app.main --date 2024-04-30
```

Run the evaluation:

```bash
python -m app.modules.evaluation.run_eval --system rule
```

Use `--system llm` to run the LLM agent instead. That one costs tokens.

Run the tests:

```bash
python -m pytest -q
```

## Project structure

```text
genai-project/
├── .gitignore
├── .env / .env.example       # my API key goes in .env, which is not committed
├── LICENSE
├── README.md
├── requirements.txt
├── pytest.ini
├── data/                     # the files we were given, plus the schedule
├── docs/workflow/            # the workflow diagram
├── app/
│   ├── main.py               # entry point
│   └── modules/
│       ├── config/           # settings and paths
│       ├── db/               # the schedule, and reading dates out of messages
│       ├── embedding/        # PDF into ChromaDB, and searching it
│       ├── agents/           # main agent, advisors, prompts, tools, memory
│       ├── fine_tuning/      # builds the training file, runs the job
│       └── evaluation/       # loads the data, scores the agent
├── streamlit_app/
│   ├── streamlit_main.py     # the Streamlit app
│   └── utils.py
├── tests/
│   ├── test_main.py
│   └── test_evals.ipynb      # accuracy and confusion matrix
└── results/                  # saved evaluation runs that the notebook reads
```

## Things I ran into

**Dates have to be worked out from the conversation date, not today's date.**
The conversations are from April 2024 and `db_Tech.sql` only fills in 2024, so
working out "next Friday" from today gives no results at all.

**There are no Mondays or Saturdays in the schedule.** `db_Tech.sql` skips
them. But people in the conversations still suggest Mondays, so the bot has to
notice that the slot does not exist and offer something else instead.

**Messages often contain two times.** Something like "this Friday at 11 AM or
next Monday at 9 AM" has to be read as two separate offers, otherwise Friday
ends up paired with the wrong time.

**The bot has to bring up the interview itself.** In the data, most `schedule`
turns come right after the candidate describes their experience, without them
asking for a time.

**The schedule had to be frozen.** `db_Tech.sql` uses `NEWID()` to decide which
slots are free, so it gives a different calendar every time it runs. I ran it
once and saved the result as `data/schedule.sqlite` so the numbers above stay
the same.

## What I could not do

**Fine-tuning.** OpenAI closed fine-tuning to new jobs while I was working on
this, so the job cannot be created:

```
403 training_not_available - "OpenAI is winding down the fine-tuning platform and
your organization is no longer able to create new fine-tuning training jobs."
```

The code for it is in `app/modules/fine_tuning/`. It builds the training file
(30 training and 14 validation examples) and uploads it fine, it just cannot
start the job. The Exit Advisor uses few-shot prompting instead. A fine-tuned
model can be plugged in later with one flag:

```bash
python -m app.modules.evaluation.run_eval --system llm --exit-model ft:...
```

**I trained on all the `end` labels, not only the "not interested" ones.** The
spec describes the Exit Advisor as spotting candidates who are not interested,
but only 4 of the 15 conversations end that way - the rest end after the
candidate accepts a time. Training on just the "not interested" ones left me
with 2 examples, which is not enough, so the advisor handles both cases.

**The folder is called `streamlit_app/`, not `streamlit/`.** If you name a
folder `streamlit`, Python imports that instead of the real Streamlit package
and the app stops working.

**No Streamlit Community Cloud deployment.** It runs locally.

## License

MIT - see [LICENSE](LICENSE).
