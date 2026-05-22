import asyncio
import re
import os
import base64
from pathlib import Path
from playwright.async_api import async_playwright

# =========================================================
# CONFIG
# =========================================================
BASE_URL = "https://normative.sumdu.edu.ua/"
SAVE_ROOT = Path(r"pdf_documents")
HEADLESS = False  # Тримаємо False, щоб Cloudflare не блокував сесію
MAX_DOWNLOADS = 100

# =========================================================
# HELPERS
# =========================================================
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:120] if len(name) > 120 else name

# =========================================================
# PARSER
# =========================================================
class SumduNormativeParser:
    def __init__(self):
        self.downloaded_count = 0

    async def start(self):
        async with async_playwright() as p:
            print("[LAUNCH] Запуск браузера...")
            browser = await p.chromium.launch(
                headless=HEADLESS,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = await browser.new_context()
            page = await context.new_page()

            print("[OPEN] Відкриваємо головну сторінку...")
            await page.goto(BASE_URL, wait_until="networkidle", timeout=120000)
            
            print("[WAIT] Очікуємо валідації Cloudflare (5 сек)...")
            await page.wait_for_timeout(5000)

            print("[PARSE] Збір структури та посилань...")
            # Збираємо елементи. Тепер ми беремо ID або унікальну частину href, щоб не залежати від версії
            items = await page.evaluate('''() => {
                const result = [];
                let currentCategory = "Загальні питання";
                const root = document.body;
                const elements = root.querySelectorAll('h1, h2, h3, h4, h5, h6, .panel-title, .category-title, a');
                
                for (const el of elements) {
                    if (el.tagName === 'A') {
                        const href = el.getAttribute('href');
                        const text = el.innerText.trim();
                        
                        if (href && href.includes('getfile')) {
                            if (text.length > 2) {
                                // Робимо URL абсолютним, щоб fetch точно знав куди стукати
                                const absoluteUrl = new URL(href, window.location.href).href;
                                result.push({
                                    category: currentCategory,
                                    text: text,
                                    url: absoluteUrl
                                });
                            }
                        }
                    } else {
                        const text = el.innerText.trim();
                        if (text.length > 3 && text.length < 150) {
                            currentCategory = text;
                        }
                    }
                }
                return result;
            }''')

            print(f"[INFO] Знайдено {len(items)} документів. Починаємо In-Browser завантаження...\n")

            for item in items:
                if self.downloaded_count >= MAX_DOWNLOADS:
                    print(f"\n[LIMIT] Досягнуто ліміт у {MAX_DOWNLOADS} файлів.")
                    break

                category = sanitize_filename(item['category'])
                doc_name = sanitize_filename(item['text'])
                file_url = item['url']

                print(f"[PROCESS] {category[:30]}... -> {doc_name[:40]}...")
                
                folder = SAVE_ROOT / category
                folder.mkdir(parents=True, exist_ok=True)

                try:
                    # ХАК РОКУ: Виконуємо ін'єкцію JS-скрипту, який робить асинхронний fetch 
                    # безпосередньо з контексту довіреної сторінки сайту.
                    # Перетворюємо бінарний файл у Base64 рядок, щоб безпечно передати його в Python.
                    b64_data = await page.evaluate('''async (url) => {
                        try {
                            const response = await fetch(url);
                            if (!response.ok) return { error: `HTTP ${response.status}` };
                            
                            const blob = await response.blob();
                            
                            // Дізнаємося розширення файлу з заголовку content-type
                            const contentType = response.headers.get('content-type') || '';
                            let ext = '.pdf';
                            if (contentType.includes('msword') || contentType.includes('document')) ext = '.doc';
                            if (contentType.includes('excel') || contentType.includes('sheet')) ext = '.xls';

                            return new Promise((resolve) => {
                                const reader = new FileReader();
                                reader.onloadend = () => resolve({ 
                                    base64: reader.result.split(',')[1],
                                    ext: ext 
                                });
                                reader.readAsDataURL(blob);
                            });
                        } catch (e) {
                            return { error: e.toString() };
                        }
                    }''', file_url)

                    # Перевіряємо, чи немає помилок виконання JS
                    if "error" in b64_data:
                        print(f"    [SKIP] Помилка завантаження через браузер: {b64_data['error']}")
                        continue

                    # Декодуємо Base64 назад у бінарний файл у коді Python
                    file_bytes = base64.b64decode(b64_data['base64'])
                    ext = b64_data['ext']
                    
                    final_path = folder / f"{doc_name}{ext}"
                    with open(final_path, "wb") as f:
                        f.write(file_bytes)

                    print(f"    [SUCCESS] Збережено: {final_path.name}")
                    self.downloaded_count += 1
                    
                    # Легка пауза, щоб не тригерити захист сервера
                    await page.wait_for_timeout(1000)

                except Exception as e:
                    print(f"    [ERROR] Непередбачувана помилка: {e}")
                    continue

            await browser.close()
            print(f"\n[DONE] Парсинг завершено! Всього збережено об'єктів: {self.downloaded_count}")

if __name__ == "__main__":
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    asyncio.run(SumduNormativeParser().start())