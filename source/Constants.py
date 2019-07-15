class Constants:
    path_resources = '../resources/'
    path_model = '../saved-model/'
    path_dataset = '../../../../../../Dataset/'
    # path_dataset = "../../../Dataset/"
    path_test_addresses = '../../../dataset_addresses/'

    file_words = path_dataset + 'words.txt'

    path_test_imgs = path_resources + 'test-imgs'

    file_char_list = path_resources + 'chars.txt'
    file_word_char_list = path_resources + 'word_chars.txt'

    file_collection_words = path_resources + 'collection_words.txt'  # will be generated in runtime

    file_collection_handwritten_words = path_resources + 'collection_handwritten_words.txt'
    # file_collection_test_address = path_resources + 'collection_test_address.txt'
    file_collection_address = path_resources + 'collection_address.txt'

    path_test_address_file = path_resources + 'addresses/'
    file_collection_home_type_1 = path_test_address_file + 'home_type_1.txt'
    file_collection_home_type_2 = path_test_address_file + 'home_type_2.txt'
    file_collection_home_type_3 = path_test_address_file + 'home_type_3.txt'
    file_collection_home_type_4 = path_test_address_file + 'home_type_4.txt'
    file_collection_company_type_1 = path_test_address_file + 'company_type_1.txt'

    file_test_img = path_resources + 'test1.png'
    file_word_beam_search = path_resources + 'word_beam_search.so'

    train_percentage = 0.85

    num_epochs = 10
    batch_size = 50
    img_size = (128, 32)  # width x height
    text_length = 32
    learning_rate = 0.0001

    decoder_word_beam = 1
    decoder_best_path = 2

    decoder_selected = decoder_word_beam
