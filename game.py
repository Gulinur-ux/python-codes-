import random

print("Salom! Raqam topish o'yiniga xush kelibsiz!")
print("Men 1 dan 100 gacha bo'lgan raqamni tanladim. Topa olasizmi?")

# Tasodifiy raqam
secret_number = random.randint(1, 100)
attempts = 0

while True:
    guess = input("Raqamingizni kiriting: ")

    # Foydalanuvchi raqam kiritganiga ishonch hosil qilish
    if not guess.isdigit():
        print("Iltimos, faqat raqam kiriting!")
        continue

    guess = int(guess)
    attempts += 1

    if guess < secret_number:
        print("Xato! Men tanlagan raqam undan kattaroq.")
    elif guess > secret_number:
        print("Xato! Men tanlagan raqam undan kichikroq.")
    else:
        print(f"Tabriklayman! Siz {attempts} urinishda to'g'ri raqamni topdingiz!")
        break
