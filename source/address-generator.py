import random

titles = ["Dr.", "Mr.", "Mrs.", "Miss.", "Prof."]
names = ["Andrew Garfield", "Peter Scott", "Joseph Dias", "Smith Roy", "Mitchelle Peters"]
roads = ["Silva Mawatha", "Olcott Mawatha", "York Street"]
sub_cities = ["Bolawalana", "Kurana", "Asgiriya", "Thalwatta"]
cities = ["Negombo", "Colombo", "Galle", "Kandy"]
companies = ["Ultimate Solutions", "One Print", "Power Tie"]
numbers = ["250", "4", "34"]
numbers_with_chars = ["250A", "4A", "34B"]
numbers_with_slash = ["250-2", "4-3", "34-B"]

count = 50

hand = Hand()


def generate_home_type_1():
    file = open("home_type_1.txt", "a+")

    for i in range(1, count):
        title = random.choice(titles)
        name = random.choice(names)
        number = random.choice(numbers)
        road = random.choice(roads)
        sub_city = random.choice(sub_cities)
        city = random.choice(cities)

        address = title + " " + name + " " + number + " " + road + " " + sub_city + " " + city + "\n"
        file.write(address)

        address_list = []
        address_list.append(title + " " + name)
        address_list.append(number + " " + road)
        address_list.append(sub_city)
        address_list.append(city)

    file.close()


def generate_home_type_2():
    file = open("home_type_2.txt", "a+")

    for i in range(1, count):
        title = random.choice(titles)
        name = random.choice(names)
        number = random.choice(numbers_with_chars)
        road = random.choice(roads)
        sub_city = random.choice(sub_cities)
        city = random.choice(cities)

        address = title + " " + name + " " + number + " " + road + " " + sub_city + " " + city + "\n"
        file.write(address)

        address_list = []
        address_list.append(title + " " + name)
        address_list.append(number + " " + road)
        address_list.append(sub_city)
        address_list.append(city)

    file.close()


def generate_home_type_3():
    file = open("home_type_3.txt", "a+")

    for i in range(1, count):
        title = random.choice(titles)
        name = random.choice(names)
        number = random.choice(numbers_with_slash)
        road = random.choice(roads)
        sub_city = random.choice(sub_cities)
        city = random.choice(cities)

        address = title + " " + name + " " + number + " " + road + " " + sub_city + " " + city + "\n"
        file.write(address)

        address_list = []
        address_list.append(title + " " + name)
        address_list.append(number + " " + road)
        address_list.append(sub_city)
        address_list.append(city)

    file.close()


def generate_home_type_4():
    file = open("home_type_4.txt", "a+")

    for i in range(1, count):
        title = random.choice(titles)
        name = random.choice(names)
        road = random.choice(roads)
        sub_city = random.choice(sub_cities)
        city = random.choice(cities)

        address = title + " " + name + " " + road + " " + sub_city + " " + city + "\n"
        file.write(address)

        address_list = []
        address_list.append(title + " " + name)
        address_list.append(road)
        address_list.append(sub_city)
        address_list.append(city)

    file.close()


def generate_company_type_1():
    file = open("company_type_1.txt", "a+")

    for i in range(1, count):
        title = random.choice(titles)
        name = random.choice(names)
        company = random.choice(companies)
        sub_city = random.choice(sub_cities)
        city = random.choice(cities)

        address = title + " " + name + " " + company + " " + sub_city + " " + city + "\n"
        file.write(address)

    file.close()


def create_image_home_type_1():
    f = open("home_type_1.txt", "r")
    f_lines = f.readlines()
    count = len(f_lines)

    for i in range(count):
        row = f_lines[i]

        words = row.split()

        print(words)

        address_list = []
        address_list.append(words[0] + " " + words[1] + " " + words[2])
        address_list.append(words[3] + " " + words[4] + " " + words[5])
        address_list.append(words[6])
        address_list.append(words[7])

        lines = address_list
        biases = [.75 for i in lines]
        style = random.randint(1, 12)
        styles = [style for i in lines]

        hand.write(
            filename='img/' + "home_type_1/" + str(i) + '.svg',
            lines=lines,
            biases=biases,
            styles=styles,
        )


def create_image_home_type_2():
    f = open("home_type_2.txt", "r")
    f_lines = f.readlines()
    count = len(f_lines)

    for i in range(count):
        row = f_lines[i]

        words = row.split()

        print(words)

        address_list = []
        address_list.append(words[0] + " " + words[1] + " " + words[2])
        address_list.append(words[3] + " " + words[4] + " " + words[5])
        address_list.append(words[6])
        address_list.append(words[7])

        lines = address_list
        biases = [.75 for i in lines]
        style = random.randint(1, 12)
        styles = [style for i in lines]

        hand.write(
            filename='img/' + "home_type_2/" + str(i) + '.svg',
            lines=lines,
            biases=biases,
            styles=styles,
        )


def create_image_home_type_3():
    f = open("home_type_3.txt", "r")
    f_lines = f.readlines()
    count = len(f_lines)

    for i in range(count):
        row = f_lines[i]

        words = row.split()

        print(words)

        address_list = []
        address_list.append(words[0] + " " + words[1] + " " + words[2])
        address_list.append(words[3] + " " + words[4] + " " + words[5])
        address_list.append(words[6])
        address_list.append(words[7])

        lines = address_list
        biases = [.75 for i in lines]
        style = random.randint(1, 12)
        styles = [style for i in lines]

        hand.write(
            filename='img/' + "home_type_3/" + str(i) + '.svg',
            lines=lines,
            biases=biases,
            styles=styles,
        )


def create_image_home_type_4():
    f = open("home_type_4.txt", "r")
    f_lines = f.readlines()
    count = len(f_lines)

    for i in range(count):
        row = f_lines[i]

        words = row.split()

        print(words)

        address_list = []
        address_list.append(words[0] + " " + words[1] + " " + words[2])
        address_list.append(words[3] + " " + words[4])
        address_list.append(words[5])
        address_list.append(words[6])

        lines = address_list
        biases = [.75 for i in lines]
        style = random.randint(1, 12)
        styles = [style for i in lines]

        hand.write(
            filename='img/' + "home_type_4/" + str(i) + '.svg',
            lines=lines,
            biases=biases,
            styles=styles,
        )


def create_image_company_type_1():
    f = open("company_type_1.txt", "r")
    f_lines = f.readlines()
    count = len(f_lines)

    for i in range(count):
        row = f_lines[i]

        words = row.split()

        print(words)

        address_list = []
        address_list.append(words[0] + " " + words[1] + " " + words[2])
        address_list.append(words[3] + " " + words[4])
        address_list.append(words[5])
        address_list.append(words[6])

        lines = address_list
        biases = [.75 for i in lines]
        style = random.randint(1, 12)
        styles = [style for i in lines]

        hand.write(
            filename='img/' + "company_type_1/" + str(i) + '.svg',
            lines=lines,
            biases=biases,
            styles=styles,
        )


generate_home_type_1()
generate_home_type_2()
generate_home_type_3()
generate_home_type_4()
generate_company_type_1()

create_image_home_type_1()
create_image_home_type_2()
create_image_home_type_3()
create_image_home_type_4()
create_image_company_type_1()
