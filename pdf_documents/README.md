# PDF Documents Storage

Цей каталог призначений для зберігання вихідних PDF-документів СумДУ, згрупованих за розділами (підрозділами).

## Структура каталогів

Створюйте підкаталоги для кожного розділу відповідно до структури вашого реєстру. Назва підкаталогу має відповідати розділу реєстру (наприклад, `2.2_academic_integrity`).

Приклад структури:
```text
pdf_documents/
├── 1.1_general_questions/
│   ├── charter.pdf
│   └── code_of_conduct.pdf
├── 2.1_quality_system/
│   └── quality_manual.pdf
└── 2.2_academic_integrity/
    ├── anti_plagiarism_regulation.pdf
    └── academic_integrity_guidelines.pdf
```

Ці PDF-файли будуть використовуватися під час парсингу, chunking-у та наповнення бази даних (`seed_db.py`).
