"""Benchmark ground truth dataset."""

BENCHMARK: list[dict] = [
    {
        "query": "Як передбачає реалізація стратегії відповідно до «Порядку денного модернізації»?",
        "expected_section_id": 19,
        "relevant_chunk_ids": [1283],
        "relevant_page_ids": [132],
    },
    {
        "query": "Які повноваження має Міністерство освіти і науки України щодо організації освітнього процесу?",
        "expected_section_id": 19,
        "relevant_chunk_ids": [321],
        "relevant_page_ids": [63],
    },
    {
        "query": "Як визначаються квоти виборних представників з числа осіб, які навчаються в університеті?",
        "expected_section_id": 19,
        "relevant_chunk_ids": [333],
        "relevant_page_ids": [63],
    },
    {
        "query": "Як передбачається використовувати споруди Науково-дослідного інституту мінеральних добрив та пігментів?",
        "expected_section_id": 19,
        "relevant_chunk_ids": [1421],
        "relevant_page_ids": [132],
    },
    {
        "query": "Як студент повинен ставитися до майна університету?",
        "expected_section_id": 20,
        "relevant_chunk_ids": [1620],
        "relevant_page_ids": [235],
    },
    {
        "query": "Які правила поведінки потрібно дотримуватися у бібліотеці та залах їдальні?",
        "expected_section_id": 20,
        "relevant_chunk_ids": [1637],
        "relevant_page_ids": [235],
    },
    {
        "query": "Як поводитися у будівлях університету щодо етикету та поведінки?",
        "expected_section_id": 20,
        "relevant_chunk_ids": [1636],
        "relevant_page_ids": [235],
    },
    {
        "query": "Як забезпечити рівність можливостей у прийомі на роботу чи навчання?",
        "expected_section_id": 20,
        "relevant_chunk_ids": [1802],
        "relevant_page_ids": [266],
    },
    {
        "query": "Який порядок діяльності Відділу університету?",
        "expected_section_id": 21,
        "relevant_chunk_ids": [2171],
        "relevant_page_ids": [313],
    },
    {
        "query": "Який є мінімальний термін публікацій наукових робіт у періодичних виданнях, що індексуються БД Scopus та/або WoS, для авторів у звітному році?",
        "expected_section_id": 21,
        "relevant_chunk_ids": [1949],
        "relevant_page_ids": [273],
    },
    {
        "query": "Які заходи здійснюються для захисту водних ресурсів та збереження морських екосистем?",
        "expected_section_id": 21,
        "relevant_chunk_ids": [2629],
        "relevant_page_ids": [350],
    },
    {
        "query": "Який зміст розділу П8.3 та П8.4 у документі?",
        "expected_section_id": 21,
        "relevant_chunk_ids": [2058],
        "relevant_page_ids": [273],
    },
    {
        "query": "Який додаток містить анкету щодо оцінювання випускниками рівня їх задоволеності якістю освітніх програм у Сумському державному університеті?",
        "expected_section_id": 22,
        "relevant_chunk_ids": [2963],
        "relevant_page_ids": [415],
    },
    {
        "query": "Хто представляє освітньо-науковий рівень вищої освіти в СумДУ?",
        "expected_section_id": 22,
        "relevant_chunk_ids": [3037],
        "relevant_page_ids": [428],
    },
    {
        "query": "Які завдання має здійснювати структура університету щодо планування освітніх програм?",
        "expected_section_id": 22,
        "relevant_chunk_ids": [2989],
        "relevant_page_ids": [422],
    },
    {
        "query": "Як відбувається внесення змін та доповнень до Положення університету?",
        "expected_section_id": 22,
        "relevant_chunk_ids": [2894],
        "relevant_page_ids": [402],
    },
    {
        "query": "Який додаток містить форму декларації про дотримання академічної доброчесності в Сумському державному університеті?",
        "expected_section_id": 23,
        "relevant_chunk_ids": [3377],
        "relevant_page_ids": [468],
    },
    {
        "query": "Яка організація структури групи щодо забезпечення академічної доброчесності?",
        "expected_section_id": 23,
        "relevant_chunk_ids": [3410],
        "relevant_page_ids": [474],
    },
    {
        "query": "Яка мета діяльності Групи щодо академічної доброчесності в Університеті?",
        "expected_section_id": 23,
        "relevant_chunk_ids": [3397],
        "relevant_page_ids": [474],
    },
    {
        "query": "Який документ регулює діяльність групи сприяння академічній доброчесності в Сумському державному університеті?",
        "expected_section_id": 23,
        "relevant_chunk_ids": [3391],
        "relevant_page_ids": [474],
    }
]
