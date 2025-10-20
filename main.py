import os
import time
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# --- Validate environment ---
if not API_KEY or not INPUT_DIR or not OUTPUT_DIR:
    raise ValueError("⛔ لطفاً متغیرهای GEMINI_API_KEY، INPUT_DIR و OUTPUT_DIR را در فایل .env تنظیم کنید.")

genai.configure(api_key=API_KEY)

# --- Constants ---
MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 5
LINES_PER_BATCH = 400


# --- Detect subtitle format ---
def detect_format(filename):
    if filename.endswith('.srt'):
        return 'srt'
    elif filename.endswith('.vtt'):
        return 'vtt'
    return None


# --- Load subtitle file ---
def load_subtitle_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# --- Build translation prompt ---
def build_translation_prompt(batch_text):
    return (
        "شما یک مترجم حرفه‌ای زیرنویس هستید. لطفاً دیالوگ‌های انگلیسی موجود در این فایل زیرنویس را به فارسی استاندارد و روان ترجمه کنید.\n"
        "در ترجمه دقت کنید:\n"
        "- فقط دیالوگ‌ها را ترجمه کنید، شماره خطوط، زمان‌ها و ساختار کلی فایل را تغییر ندهید.\n"
        "- جملات را طوری ترجمه کنید که برای مخاطب فارسی‌زبان طبیعی به نظر برسند.\n"
        "- اگر اصطلاح، محاوره یا فرهنگ خاصی وجود دارد، آن را به درستی بازتولید کنید.\n"
        "- از ترجمهٔ لفظی و مستقیم خودداری کنید.\n"
        "- طول جملات را مناسب با زمان‌های زیرنویس حفظ کنید (بهتر است خیلی طولانی نباشند).\n\n"
        f"{batch_text.strip()}\n\n"
        "---\n"
        "فقط فایل زیرنویس با دیالوگ‌های ترجمه شده، بدون هیچ توضیح یا اضافه‌ای."
    )

# --- Translate a batch ---
async def translate_batch(batch_lines):
    batch_text = "\n".join(batch_lines)
    prompt = build_translation_prompt(batch_text)

    generation_config = genai.GenerationConfig(
        max_output_tokens=8192,
        temperature=0.2
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            chat = genai.GenerativeModel(MODEL_NAME).start_chat()
            print(chat.send_message(prompt, generation_config=generation_config))
            response = chat.send_message(prompt, generation_config=generation_config)
            translated_text = response.text.strip()
            return translated_text.splitlines()
        except Exception as e:
            print(f"❌ خطا در تلاش {attempt} برای ترجمه batch: {str(e)}")
            if "429" in str(e) or "quota" in str(e).lower():
                await asyncio.sleep(3)
            else:
                await asyncio.sleep(5)
    print("⛔ ترجمه batch ناموفق بود.")
    return batch_lines  # Return original if failed


# --- Save translated file ---
def save_translated_file(original_file_path, input_root, output_root, translated_lines):
    relative_path = os.path.relpath(original_file_path, input_root)
    base, ext = os.path.splitext(relative_path)

    if "_en" in base:
        new_base = base.replace("_en", "")
    else:
        new_base = base + ""

    output_file = new_base + ext
    output_path = os.path.join(output_root, output_file)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(translated_lines))
    print(f"✅ فایل ذخیره شد: {output_path}")


# --- Process subtitle file ---
async def process_file(file_path, input_root, output_root):
    fmt = detect_format(file_path)
    if not fmt:
        print(f"⛔ فرمت پشتیبانی نمی‌شود: {file_path}")
        return

    print(f"\n📄 در حال پردازش فایل: {file_path}")
    original_text = load_subtitle_text(file_path)
    lines = original_text.splitlines()

    total_lines = len(lines)
    print(f"📚 تعداد کل خطوط: {total_lines}")

    translated_lines = []

    for i in range(0, total_lines, LINES_PER_BATCH):
        batch = lines[i:i + LINES_PER_BATCH]
        print(f"🔄 ترجمه batch {i // LINES_PER_BATCH + 1}: {len(batch)} خط")
        translated_batch = await translate_batch(batch)
        translated_lines.extend(translated_batch)
        await asyncio.sleep(2)

    save_translated_file(file_path, input_root, output_root, translated_lines)


# --- Main execution ---
async def main():
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(('.srt', '.vtt')):
                full_path = os.path.join(root, file)
                await process_file(full_path, INPUT_DIR, OUTPUT_DIR)
                await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())

