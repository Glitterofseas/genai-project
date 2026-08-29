# Source materials

Everything supplied for this project, where it now lives, and proof it is
unchanged. All files below are byte-identical to what was provided; only their
names and locations changed, so that the repository follows the structure the
assignment requires.

## Documents

| Supplied as | Now at | Bytes |
|---|---|---|
| `Python Developer Job Description.pdf` | `data/Python Developer Job Description.pdf` | 99,900 |
| `sms_conversations.json` | `data/sms_conversations.json` | 26,957 |
| `db_Tech.sql` | `data/db_Tech.sql` | 2,018 |
| `README.md.txt` | `docs/README_template_original.txt` | 3,483 |

`README.md.txt` was the blank project template. It is kept here for reference;
the filled-in project documentation is `README.md` at the repository root.

## Assignment specification (GenAI Project.pdf, 5 pages)

The PDF itself was never on disk - it was supplied as screenshots. Renamed by
content, since the capture filenames carried no meaning.

| Supplied as | Now at | Bytes |
|---|---|---|
| `Screenshot 2026-08-29 170443.png` | `docs/spec/01_project_overview.png` | 77,515 |
| `Screenshot 2026-08-29 170536.png` | `docs/spec/02_data_and_resources.png` | 177,277 |
| `Screenshot 2026-08-29 170543.png` | `docs/spec/03_project_structure.png` | 112,086 |
| `Screenshot 2026-08-29 170551.png` | `docs/spec/04_additional_implementation_steps.png` | 149,701 |
| `Screenshot 2026-08-29 170558.png` | `docs/spec/05_main_components.png` | 61,122 |

Two earlier screenshots, `inst 1.png` and `inst2.png`, were present at the start
of the session and were replaced in the folder by the five above before they
could be moved. They are not lost content: `inst 1.png` was the Project Overview
page and `inst2.png` was the Main Components page, preserved here as
`01_project_overview.png` and `05_main_components.png` respectively.

## Workflow diagram ("One Turn in the Conversation")

Four screenshots of one diagram. Renamed into **reading order**, which differs
from capture order - the diagram was captured top, bottom, middle, then the
routing section.

| Supplied as | Now at | Bytes |
|---|---|---|
| `workflow/Screenshot 2026-08-29 171833.png` | `docs/workflow/01_entry_and_legend.png` | 36,598 |
| `workflow/Screenshot 2026-08-29 171932.png` | `docs/workflow/02_main_agent_routing.png` | 32,776 |
| `workflow/Screenshot 2026-08-29 171906.png` | `docs/workflow/03_advisor_internals.png` | 55,654 |
| `workflow/Screenshot 2026-08-29 171848.png` | `docs/workflow/04_convergence_and_exit.png` | 30,495 |

## Verifying

```bash
python tools/verify_sources.py
```

Checks every file above still exists at its recorded size and still parses -
the PDF opens, the JSON loads all 15 conversations, the SQL retains its
`CREATE DATABASE`, and each PNG has a valid header.
