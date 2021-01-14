swear_words = set([])
hate_words = set([])
gender_words = set([])
race_words = set([])
origin_words = set([])
disability_words = set([])
religion_words = set([])
orientation_words = set([])
ethnicity_words = set([])

def parse_files():
    global swear_words,hate_words,gender_words,race_words,origin_words,disability_words,religion_words,orientation_words,ethnicity_words
    hw =[]
    file1 = open("hate.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    hate_words = set(hw)
    file1.close()
    hw =[]
    file1 = open("disability.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    disability_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("ethnicity.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    ethnicity_words = set(hw)
    file1.close()
    hw =[]
    file1 = open("gender.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    gender_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("origin.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    origin_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("race.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    race_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("religion.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    religion_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("sexual.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    orientation_words= set(hw)
    file1.close()
    hw =[]
    file1 = open("swear.txt","r", encoding='utf-8')
    for line in file1:
        hw = hw + [line.lower().rstrip()]
    print(hw)
    swear_words= set(hw)
    file1.close()

parse_files()

print(swear_words)