import gzip, html, json, os, re, requests
# from dataset import amazon_dataset2fullname
from html.parser import HTMLParser
from tqdm import tqdm
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed


amazon_dataset2fullname = {
    # 'Beauty': 'All_Beauty',
    'Beauty': 'Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games'
}
def preprocess_img_dict(rating_inters, dataset, input_path, output_path): # main function
    file_prefix = os.path.join(output_path, dataset)
    os.makedirs(file_prefix, exist_ok=True)

    file_path = os.path.join(file_prefix, f"imgURL_{dataset}_5.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    rating_users, rating_items = get_user_item_from_ratings(rating_inters)
    print(">> preprocess_img_dict")
    # from IPython import embed ; embed()
    # load item text and clean
    item_metadata_dict, meta_item_set = build_item_imUrl(rating_items, dataset, input_path)
    print(">> build_item_imUrl")
    
    for asin in list(rating_items - meta_item_set):
        item_metadata_dict[asin] = {}
    print(">> item_img_dic[asin]")

    file_prefix = os.path.join(output_path, f"imgURL_{dataset}")
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(item_metadata_dict, f, indent=4, ensure_ascii=False)
    
    return item_metadata_dict


def get_user_item_from_ratings(ratings):
    users, items = set(), set()
    for line in ratings:
        user, item, rating, time = line
        users.add(user)
        items.add(item)

    return users, items


# def build_clean_item_metadata(target_items, dataset, input_path):
#     item_text_dict = {}
#     processed_items = set()
#     dataset_full_name = amazon_dataset2fullname[dataset]
#     # if dataset in ["Beauty"]
#     meta_file_path = os.path.join(input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
#     meta_item_set= set()

#     with gzip.open(meta_file_path, 'r') as fp:
#         for line in tqdm(fp):
#             try:
#                 data = json.loads(line)
#             except:
#                 ## for beauty, sports, toys
#                 line_str = line.decode('utf-8')
#                 data = ast.literal_eval(line_str)

#             item = data['asin']

#             if item in target_items and item not in processed_items:
#                 from IPython import embed ;embed()
#                 processed_items.add(item)

#                 item_text_dict[item] = dict()
#                 for key in data.keys():
#                     if key in ['tech1', 'tech2', 'fit', 'similar_item', 'also_buy', 'also_view', 'imageURL', 'imageURLHighRes', 'date']:
#                         # table style: tech1, tech2, fit, similar_item
#                         # list of item id: also_buy, also_view, similar_item
#                         # URL: imageURL, imageURLHighRes
#                         # Date: date
#                         continue
#                         # example of details: {'Shipping Weight:': '2.4 ounces (', 'ASIN:': 'B01HDXZR5E'}
#                     cleaned_text = clean_text(data[key])
#                     if contains_html(cleaned_text):
#                         cleaned_text = normalize_spaces(strip_html(cleaned_text))
#                     item_text_dict[item][key] = cleaned_text

#                 meta_item_set.add(item)

#     return item_text_dict, meta_item_set

def build_item_imUrl(target_items, dataset, input_path):
    item_url_dict = {}
    processed = set()
    dataset_full = amazon_dataset2fullname[dataset]
    # if dataset in ["Beauty"]
    meta_path = os.path.join(input_path, 'Metadata', f'meta_{dataset_full}.json.gz')
    meta_item_set= set()
    
    print(">> Bulid_item_imURL")
    with gzip.open(meta_path, 'r') as fp:
        for line in fp:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = ast.literal_eval(line.decode('utf-8'))

            asin = data.get('asin')
            if asin in target_items and asin not in processed:
                processed.add(asin)
                
                item_url_dict[asin] = dict()
                # imUrl 키만 뽑아서 저장
                url = data.get('imUrl')
                if url:
                    item_url_dict[asin]["url"] = url
                
                meta_item_set.add(asin)

            # 모든 타겟 아이템을 다 찾으면 중단할 수도 있음
            if processed == set(target_items):
                break

    return item_url_dict, meta_item_set


def clean_text(raw_text):
    if isinstance(raw_text, list):
        # for amazon review 2018
        # cleaned_text = ' '.join(raw_text)
        # for amazon review 2014
        cleaned_text = str(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    elif isinstance(raw_text, float):
        cleaned_text = str(raw_text)
    elif isinstance(raw_text, int):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text

    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1

    if index != 0:
        cleaned_text = cleaned_text[:index]
    
    return cleaned_text


def normalize_spaces(text):
    return re.sub(r'\s{2,}', ' ', text)


class HTMLDetector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found_tag = False

    def handle_starttag(self, tag, attrs):
        self.found_tag = True


def contains_html(text):
    parser = HTMLDetector()
    parser.feed(text)
    return parser.found_tag


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_data(self):
        return ''.join(self.text_parts)


def strip_html(text):
    stripper = HTMLStripper()
    stripper.feed(text)
    return stripper.get_data()


# def generate_item_img_dict(item_metadata_dict, selected_features, dataset, output_path): # main function
#     item_metatext_dict = {}
#     # path = "/data/dongcheol/processed/Beauty/img"
#     path = os.path.join(output_path, dataset, "img")
#     for item_id, features in tqdm(item_metadata_dict.items()):
#         item_metatext_dict[item_id] = ""
        
#         imgURL = features.get('url')
        
#         save_path = os.path.join(path, f"{item_id}.jpg")
        
#         if os.path.exists(save_path):
#             continue
#         try:
#             resp = requests.get(imgURL, timeout=5, stream=True)
#             resp.raise_for_status()
#             with open(save_path, 'wb') as f:
#                 for chunk in resp.iter_content(1024):
#                     f.write(chunk)
#         except Exception as e:
#             print(f"[Error] {item_id} 다운로드 실패: {e}")
            
#         item_metatext_dict[item_id] = save_path

#     return item_metatext_dict

def generate_item_img_dict(item_metadata_dict, selected_features, dataset, output_path,
                           max_workers=16, chunk_size=8192):
    """
    item_metadata_dict: {item_id: { ..., 'url': imgURL, ... }, ...}
    selected_features: (unused here, kept for interface compatibility)
    dataset: str, ex. "Beauty"
    output_path: base dir where images/ will be created
    max_workers: 동시 다운로드 스레드 수
    chunk_size: 스트리밍 시 읽어올 청크 크기
    """
    # 1) 출력 디렉토리 준비
    save_dir = os.path.join(output_path, dataset, "img")
    os.makedirs(save_dir, exist_ok=True)

    # 2) requests.Session 생성하여 함수 속성으로 붙여두기
    session = requests.Session()
    def _download_one(item_id, url):
        ext = url.split('?')[0].split('.')[-1]
        save_path = os.path.join(save_dir, f"{item_id}.{ext}")
        if os.path.exists(save_path):
            return item_id, save_path, True

        try:
            resp = session.get(url, timeout=5, stream=True)
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size):
                    if chunk:
                        f.write(chunk)
            return item_id, save_path, True
        except Exception:
            return item_id, None, False

    # 3) ThreadPoolExecutor 로 병렬 다운로드
    results = {}
    args = [
        (item_id, features.get('url'))
        for item_id, features in item_metadata_dict.items()
        if features.get('url')
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(_download_one, item_id, url): item_id
            for item_id, url in args
        }
        for fut in tqdm(as_completed(future_to_item),
                        total=len(future_to_item),
                        desc="Downloading images"):
            item_id = future_to_item[fut]
            _, save_path, success = fut.result()
            if success:
                results[item_id] = save_path
            else:
                # 실패한 경우 None 또는 빈 문자열으로 표시
                results[item_id] = None

    return results


def get_unique_features(item_metadata_dict):
    unique_features = set()
    for item_id, features in item_metadata_dict.items():
        unique_features.update(features.keys())

    return unique_features


def get_unique_words_for_features(item_metadata_dict, feature_list, dataset):
    unique_words = set()

    for _, features in item_metadata_dict.items():
        for feature, feature_text in features.items():
            if feature not in feature_list:
                continue
            
            if feature == 'category': ## no category data in Pantry
                if dataset in ["Scientific"]:
                    words = re.findall(r'\b\w+\b', feature_text)
                    unique_words.update(word.lower() for word in words)
                elif isinstance(feature_text, str):  # typical for text fields
                    words = feature_text.lower().split()  # simple tokenization
                    unique_words.update(words)
                elif isinstance(feature_text, list):  # for list-based fields like categories
                    for item in feature_text:
                        if isinstance(item, list): # Nested list (e.g., [["Beauty", "Skincare"]])
                            unique_words.update(sub.lower() for sub in item)
                        else:
                            unique_words.update(item.lower())
            else:
                if isinstance(feature_text, str):  # typical for text fields
                    words = feature_text.lower().split()  # simple tokenization
                    unique_words.update(words)
                elif isinstance(feature_text, list):  # for list-based fields like categories
                    for item in feature_text:
                        if isinstance(item, list): # Nested list (e.g., [["Beauty", "Skincare"]])
                            unique_words.update(sub.lower() for sub in item)
                        else:
                            unique_words.update(item.lower())
                elif isinstance(feature_text, (int, float)): # convert number to string
                    unique_words.update(str(feature_text))
    
    return unique_words


def get_word2id_dict(unique_words):
    word2id_dict = {word: idx for idx, word in enumerate(sorted(unique_words))}

    return word2id_dict


