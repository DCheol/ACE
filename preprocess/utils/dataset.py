import collections, gzip, json, os, random, argparse, csv
import numpy as np

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
def preprocess_rating(dataset, input_path): # main function
    """
    Load Rating Matrix of Dataset \n
    [input] dataset, input_path \n
    [output] rating_inters = (user, item, rating, timestamp) (sorted)
    """
    dataset_full_name = amazon_dataset2fullname[dataset]
    print('Load Rating Matrix of Dataset: ', dataset_full_name)

    # load ratings
    rating_file_path = os.path.join(input_path, '5cores', f'{dataset_full_name}_5.json.gz')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    print('Raw Data Info')
    print('# Ratings: ', len(rating_inters), ' # Users: ', len(rating_users), " # Items:", len(rating_items))
    
    # Filter out users and items with less than k interactions
    # rating_inters = filter_inters(rating_inters, user_k_core_threshold=args.user_k, item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)

    return rating_inters


def load_ratings(file):
    users, items, inters = set(), set(), set()

    with gzip.open(file, 'r') as fp:
        for line in fp:
            data = json.loads(line)
            user, item, rating, time = data["reviewerID"], data['asin'], data["overall"], data['unixReviewTime']

            users.add(user)
            items.add(item)
            inters.add((user, item, float(rating), int(time)))

    return sorted(users), sorted(items), inters


def filter_inters(inters, user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []

    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(item2count, item_k_core_threshold)

            if n_filtered_users == 0 and n_filtered_items == 0:
                break

            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1

            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('Epoch %d The number of inters: %d, users: %d, items: %d' % (idx, len(inters), len(user2count), len(item2count)))

    return inters


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    valid_object = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            valid_object.add(unit)
    return valid_object, len(unit2count) - len(valid_object)


def make_inters_in_order(inters):
    new_inters = list()

    for inter in inters:
        user, item, rating, timestamp = inter
        new_inters.append((user, item, rating, timestamp))

    new_inters.sort(key=lambda x: x[1])
    new_inters.sort(key=lambda x: x[3])
    new_inters.sort(key=lambda x: x[0])

    return new_inters


def object2dict(rating_inters, dataset, output_path): # main function
    """
    save order dictions at output_path
    item2index: {item_id}\t{index}\n
    user2index: {user_id}\t{index}
    """
    file_prefix = os.path.join(output_path, dataset)

    if os.path.exists(os.path.join(file_prefix, f"{dataset}.item2index")) and os.path.exists(os.path.join(file_prefix, f"{dataset}.user2index")):
        print("Exist uid/iid file")
        item2index = load_index_file(os.path.join(file_prefix, f"{dataset}.item2index"))
        user2index = load_index_file(os.path.join(file_prefix, f"{dataset}.user2index"))
    else:
        print("no Exist uid/iid file")
        _, user2index, item2index = convert_inters2dict(rating_inters)
        os.makedirs(file_prefix, exist_ok=True)

        with open(os.path.join(file_prefix, f"{dataset}.item2index"), "w") as file:
            for item_id, index in item2index.items():
                file.write(f"{item_id}\t{index}\n")
        with open(os.path.join(file_prefix, f"{dataset}.user2index"), "w") as file:
            for user_id, index in user2index.items():
                file.write(f"{user_id}\t{index}\n")

    return user2index, item2index


def load_index_file(path):
    object2index = dict()
    with open(path, "r") as file:
        for line in file:
            object_id = line.strip().split("\t")[0]
            index = line.strip().split("\t")[1]
            object2index[object_id] = index
            # object2index[int(object_id)] = int(index)
    return object2index


def convert_inters2dict(inters): # Generate Sequential Data
    user2items = collections.defaultdict(list) # Sequential Data
    user2index, item2index = dict(), dict()

    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
            
        user2items[user2index[user]].append(item2index[item])

    print(f"convert_inters2dict")
    # from IPython import embed ; embed()
    return user2items, user2index, item2index


def save_as_recbole_general_inter(rating_inters, dataset, output_path):
    """
    save rating inters as recbole dataset (.inter) at dataset/output_path
    """
    os.makedirs(os.path.join(output_path, dataset), exist_ok=True)
    with open(os.path.join(output_path, dataset, f'{dataset}.inter'), 'w') as file:
        file.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        for uid, iid, rate, time in rating_inters:
            file.write(f'{uid}\t{iid}\t{rate}\t{time}\n')


def load_atomic_inter_file(inter_path):
    """Load a RecBole-style .inter file and return the header plus rows."""
    with open(inter_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        rows = [row for row in reader if row]
    return header, rows


def build_id_map_from_inter(
    inter_path,
    output_path=None,
    user_k_core_threshold=0,
    item_k_core_threshold=0,
):
    """
    Build an id_map.json-compatible dictionary from a RecBole .inter file.

    The mapping follows first-seen order in the interaction file and uses
    1-based ids for users/items, matching the existing project convention.
    """
    _, rows = load_atomic_inter_file(inter_path)

    rating_inters = []
    for row in rows:
        if len(row) < 4:
            continue
        user_id, item_id, rating, timestamp = row[:4]
        rating_inters.append((user_id, item_id, float(rating), int(float(timestamp))))

    if user_k_core_threshold or item_k_core_threshold:
        rating_inters = filter_inters(
            rating_inters,
            user_k_core_threshold=user_k_core_threshold,
            item_k_core_threshold=item_k_core_threshold,
        )

    user2id = {}
    item2id = {}
    for user_id, item_id, _, _ in rating_inters:
        if user_id not in user2id:
            user2id[user_id] = len(user2id) + 1
        if item_id not in item2id:
            item2id[item_id] = len(item2id) + 1

    id2user = {str(idx): token for token, idx in user2id.items()}
    id2item = {str(idx): token for token, idx in item2id.items()}
    id_map = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
        "attribute2id": {},
        "id2attribute": {},
        "attributeid2num": {},
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=2)

    return id_map


def seed_everything(seed=42):
    random.seed(seed)                        # Python random
    np.random.seed(seed)                     # NumPy
    os.environ["PYTHONHASHSEED"] = str(seed) # Python hash-based ops

    try:
        import torch
        torch.manual_seed(seed)              # PyTorch CPU
        torch.cuda.manual_seed(seed)         # PyTorch GPU
        torch.cuda.manual_seed_all(seed)     # Multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed — no problem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Scientific', help='Pantry / Scientific / Instruments / Arts / Office')
    parser.add_argument('--input_path', type=str, default='raw_data')
    parser.add_argument('--output_path', type=str, default='/data/jaewan/processed/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    # parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    # parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    # parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()

