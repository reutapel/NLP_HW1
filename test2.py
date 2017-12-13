import string
word = 'De.'
invalidChars = set(string.punctuation)


if (not word.islower() and not word.isupper() or word.isupper()) and (word not in invalidChars and not word.isdigit()):
    print(True)