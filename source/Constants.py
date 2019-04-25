class Constants:
    path_resources = '../resources/'
    path_model = '../saved-model/'
    path_dataset = "../../../../../../Dataset/"
    # path_dataset = "../../../Dataset/"
    
    file_char_list = path_resources + 'chars.txt'
    file_test_img = path_resources + 'test0.png'

    epochs = 2
    batch_size = 50
    img_size = (128, 32)  # width x height
    text_length = 32
    learning_rate = 0.0001
