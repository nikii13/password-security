"""
Word lists used by feature extraction and by the challenge-set generator.

IMPORTANT (methodology): if the challenge passwords were built only from the
same words the feature extractor searches for, the ML model would detect
them trivially and the result would be circular. So the challenge vocabulary
is split in two:

  CHALLENGE_WORDS_SEEN    -> also in COMMON_WORDS (the feature dictionary)
  CHALLENGE_WORDS_UNSEEN  -> NOT in COMMON_WORDS

In the evaluation, report results for both halves separately. If the model
also flags the "unseen" ones, it is detecting the *structure*
(Word + year + symbol), not just memorised words.
"""

SEASONS = ["spring", "summer", "autumn", "winter",
           "primavera", "verano", "otono", "invierno"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december",
          "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
COMMON_ENGLISH = ["password", "love", "iloveyou", "welcome", "hello", "monkey",
                  "dragon", "master", "sunshine", "princess", "football",
                  "baseball", "soccer", "shadow", "superman", "batman", "angel",
                  "freedom", "secret", "letmein", "admin", "login", "qwerty",
                  "flower", "happy", "lucky", "money", "computer", "internet",
                  "baby", "girl", "boy", "friend", "family", "forever", "heart",
                  "star", "tiger", "cookie", "chocolate", "purple", "orange",
                  "banana", "cheese", "pepper", "summer", "jesus", "michael"]
COMMON_SPANISH = ["contrasena", "hola", "amor", "teamo", "casa", "perro",
                  "gato", "futbol", "familia", "amigo", "corazon", "bonita",
                  "princesa", "estrella", "sol", "luna"]
COMMON_NAMES = ["maria", "jose", "juan", "carlos", "ana", "laura", "david",
                "daniel", "jessica", "ashley", "jennifer", "michelle", "andrew",
                "joshua", "matthew", "nicole", "sarah", "thomas", "robert",
                "william", "anna", "pablo", "lucia", "sofia"]
CITIES_SEEN = ["madrid", "london", "paris", "barcelona", "newyork"]
TEAMS_SEEN = ["realmadrid", "barca", "chelsea", "arsenal", "liverpool"]

# Dictionary the feature extractor uses ("contains common word")
COMMON_WORDS = sorted(set(SEASONS + MONTHS + COMMON_ENGLISH + COMMON_SPANISH
                          + COMMON_NAMES + CITIES_SEEN + TEAMS_SEEN))

# Challenge vocabulary
CHALLENGE_WORDS_SEEN = ["summer", "winter", "spring", "september", "welcome",
                        "password", "football", "maria", "madrid", "london",
                        "chelsea", "dragon", "sunshine", "carlos", "princess"]
CHALLENGE_WORDS_UNSEEN = ["reykjavik", "valencia", "sevilla", "berlin",
                          "atletico", "valur", "manchester", "elena",
                          "javier", "kristin", "gudrun", "pizza", "coffee",
                          "guitar", "rainbow"]

assert not set(CHALLENGE_WORDS_UNSEEN) & set(COMMON_WORDS), \
    "unseen challenge words must not be in the feature dictionary"

# Keyboard rows for keyboard-walk detection (qwerty layout)
KEYBOARD_ROWS = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
