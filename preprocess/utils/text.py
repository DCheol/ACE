import gzip, html, json, os, re, csv
# from dataset import amazon_dataset2fullname
from html.parser import HTMLParser
import ast
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
def preprocess_text_dict(rating_inters, dataset, input_path, output_path): # main function
    file_prefix = os.path.join(output_path, dataset)
    os.makedirs(file_prefix, exist_ok=True)

    file_path = os.path.join(file_prefix, f"{dataset}_5.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    rating_users, rating_items = get_user_item_from_ratings(rating_inters)

    # load item text and clean
    item_metadata_dict, meta_item_set = build_clean_item_metadata(rating_items, dataset, input_path)
    for asin in list(rating_items - meta_item_set):
        item_metadata_dict[asin] = {}

    file_prefix = os.path.join(output_path, dataset)
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


def build_clean_item_metadata(target_items, dataset, input_path):
    item_text_dict = {}
    processed_items = set()
    dataset_full_name = amazon_dataset2fullname[dataset]
    # if dataset in ["Beauty"]
    meta_file_path = os.path.join(input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    meta_item_set= set()

    with gzip.open(meta_file_path, 'r') as fp:
        for line in fp:
            try:
                data = json.loads(line)
            except:
                ## for beauty, sports, toys
                line_str = line.decode('utf-8')
                data = ast.literal_eval(line_str)

            item = data['asin']

            if item in target_items and item not in processed_items:
                processed_items.add(item)

                item_text_dict[item] = dict()
                for key in data.keys():
                    if key in ['tech1', 'tech2', 'fit', 'similar_item', 'also_buy', 'also_view', 'imageURL', 'imageURLHighRes', 'date']:
                        # table style: tech1, tech2, fit, similar_item
                        # list of item id: also_buy, also_view, similar_item
                        # URL: imageURL, imageURLHighRes
                        # Date: date
                        continue
                        # example of details: {'Shipping Weight:': '2.4 ounces (', 'ASIN:': 'B01HDXZR5E'}
                    cleaned_text = clean_text(data[key])
                    if contains_html(cleaned_text):
                        cleaned_text = normalize_spaces(strip_html(cleaned_text))
                    item_text_dict[item][key] = cleaned_text

                meta_item_set.add(item)

    return item_text_dict, meta_item_set


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


def generate_item_metatext_dict(item_metadata_dict, selected_features): # main function
    item_metatext_dict = {}

    for item_id, features in item_metadata_dict.items():
        item_metatext_dict[item_id] = ""
        for feature in selected_features[:-1]:
            if feature not in features:
                item_metatext_dict[item_id] += f"{feature}: None ; "
            else:
                item_metatext_dict[item_id] += f"{feature}: {item_metadata_dict[item_id][feature]} ; "
        
        if selected_features[-1] not in features:
            item_metatext_dict[item_id] += f"{selected_features[-1]}: None"
        else:
            item_metatext_dict[item_id] += f"{selected_features[-1]}: {item_metadata_dict[item_id][selected_features[-1]]}"

    return item_metatext_dict


def _clean_item_str_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v not in (None, ""))
    if isinstance(value, dict):
        return str(value)
    return str(value)


def _pretty_field_name(field_name):
    return field_name.split(":", 1)[0].strip()


def build_item_str_from_atomic_item(
    item_path,
    output_path=None,
    dataset_name=None,
    fields=None,
    item_filter=None,
):
    """
    Convert a RecBole .item file into a {item_id: text} dictionary.

    The default output is a compact natural-language summary built from the
    non-id fields that appear in the .item header.
    """
    item_str_dict = {}

    with open(item_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise ValueError(f"No header found in item file: {item_path}")

        item_field = header[0]
        if fields is None:
            fields = [field for field in header[1:]]

        dataset_label = dataset_name if dataset_name else os.path.basename(item_path).split(".")[0]
        field_index = {name: idx for idx, name in enumerate(header)}

        for row in reader:
            if not row:
                continue
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]

            item_idx = field_index.get(item_field)
            if item_idx is None:
                continue
            item_id = row[item_idx].strip()
            if not item_id:
                continue
            if item_filter is not None and item_id not in item_filter:
                continue

            parts = []
            for field in fields:
                field_idx = field_index.get(field)
                if field_idx is None:
                    continue
                value = _clean_item_str_value(row[field_idx].strip())
                if value == "":
                    continue
                # parts.append(f"{_pretty_field_name(field)} is {value}")
                parts.append(f"{_pretty_field_name(field)}: {value}")

            if parts:
                item_str_dict[item_id] = (
                    f"The {dataset_label} item has the following attributes: \n "
                    + "; ".join(parts)
                    + ";"
                    
                    # "; ".join(parts)
                )
            else:
                item_str_dict[item_id] = f"The {dataset_label} item has no available attributes."

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(item_str_dict, f, ensure_ascii=False, indent=2)

    return item_str_dict


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
