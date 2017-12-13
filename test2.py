import string
word = 'Teamsters'
invalidChars = set(string.punctuation)

if (not word.islower() and not word.isupper() or word.isupper()) \
     and (word not in invalidChars and not word.isdigit()):
    print(True)