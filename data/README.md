# Data placement

The challenge data contains anonymized student submissions and remains private to the course group.

Place the distributed files under `data/raw/` using the layout supplied with the challenge. The JSON records reference source files under `submissions/<problem-id>/<problem-id>-<student-id>.cpp`; the loader must resolve those references without copying private code into tracked files.

Expected future destinations:

- `data/raw/`: original challenge files; ignored by Git.
- `data/processed/`: normalized records and manifests; ignored by Git.
- `data/splits/`: safe split IDs if the team is permitted to share them.
- `data/cache/`: local preprocessing/retrieval cache; ignored by Git.

Do not commit raw submissions, private test data, manually labeled test examples, or student-identifying information.
