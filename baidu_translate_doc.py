import requests
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def translate_text(text, source_lang, target_lang, app_id, app_key):
    url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    payload = {
        'q': text,
        'from': source_lang,
        'to': target_lang,
        'appid': app_id,
        'salt': '123456',  # 随机数，这里简单用固定值
        'sign': '',  # 签名字段，稍后计算
    }
    payload['sign'] = generate_sign(payload, app_key)

    response = requests.get(url, params=payload)
    if response.status_code == 200:
        try:
            return response.json()['trans_result'][0]['dst']
        except KeyError:
            print('Error: Failed to parse translation response.')
            return None
    else:
        print('Error: Translation request failed with status code', response.status_code)
        return None


def generate_sign(payload, app_key):
    import hashlib

    sign_str = payload['appid'] + payload['q'] + payload['salt'] + app_key
    md5 = hashlib.md5()
    md5.update(sign_str.encode('utf-8'))
    return md5.hexdigest()


def translate_pdf(pdf_path, source_lang, target_lang, app_id, app_key):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        output_path = pdf_path[:-4] + '_translated.pdf'
        c = canvas.Canvas(output_path, pagesize=letter)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            translated_text = translate_text(page_text, source_lang, target_lang, app_id, app_key)
            page_text_translated = translated_text if translated_text else page_text

            # Add the translated text to the new PDF page
            c.drawString(72, 720, page_text_translated)  # You may need to adjust the position based on your PDF

            # Add a new page for the next iteration
            if page_num < num_pages - 1:
                c.showPage()

        c.save()


if __name__ == '__main__':
    # 请替换以下信息为你自己的API密钥和需要翻译的PDF文件路径
    APP_ID = '20230801001765278'
    APP_KEY = '0TiiuAWpzP2dwrkgXuFE'
    SOURCE_LANG = 'en'
    TARGET_LANG = 'zh'
    PDF_FILE_PATH = 'input.pdf'

    translate_pdf(PDF_FILE_PATH, SOURCE_LANG, TARGET_LANG, APP_ID, APP_KEY)
