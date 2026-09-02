# SMS Recruiting Chatbot - GenAI Project

## Project purpose

An SMS chatbot that talks to candidates applying for a Python Developer job.
It answers their questions and then either books them an interview or ends the
conversation politely.

A **Main Agent** runs the conversation. On every turn it picks one of three
actions - `continue`, `schedule` or `end` - and asks three **advisor agents**
for help before deciding:

| Advisor | Decides | Uses |
|---|---|---|
| Exit Advisor | should we end the conversation? | - |
| Scheduling Advisor | should we offer an interview time? | the SQL schedule |
| Info Advisor | does the candidate need info about the job? | the PDF in ChromaDB |

The Main Agent asks one advisor at a time and only asks another if it still
needs to, following the workflow diagram in `docs/workflow/`. SQL is only
queried if the Scheduling Advisor says yes, and ChromaDB only if the Info
Advisor says yes.

Built with LangChain, OpenAI, ChromaDB, SQL and Streamlit.

## How to install and run locally

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

Build the ChromaDB index once. The schedule is already included, so it does not
need building.

```bash
python -m app.modules.embedding.build_index
```

## Basic usage examples

Run the Streamlit app:

```bash
streamlit run streamlit_app/streamlit_main.py
```

Fill in the registration form (or skip it) and chat as the candidate. The
sidebar switches between the LLM agent and a free rule-based one, and the trace
under each reply shows which advisors were asked.

Chat in the terminal instead:

```bash
python -m app.main --date 2024-04-30
```

Run the evaluation. `--system llm` uses the LLM agent and costs tokens:

```bash
python -m app.modules.evaluation.run_eval --system rule
```

Run the tests:

```bash
python -m pytest -q
```

Open `tests/test_evals.ipynb` for the accuracy scores and confusion matrices.

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

## Fine-tuning

The spec asks for the Exit Advisor to be fine-tuned. The training and validation
files are built and committed (`data/fine_tuning/`), but the job cannot be
created - OpenAI has wound down self-serve fine-tuning:

> OpenAI is winding down the fine-tuning platform and your organization is no
> longer able to create new fine-tuning training jobs.

So the Exit Advisor uses few-shot prompting instead. If the account ever regains
access, the job still launches with:

```bash
python -m app.modules.fine_tuning.run_job --launch
```

## License

MIT - see [LICENSE](LICENSE).
