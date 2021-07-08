import random

emotion_map = {
    1: [":slightly_smiling_face:"],  # neutral
    2: [":smiling_face_with_halo:"],  # calm
    3: [":beaming_face_with_smiling_eyes:"],  # happy
    4: [":pensive_face:"],  # sad
    5: [":pouting_face:"],  # angry
    6: [":fearful_face:"],  # fearful
    7: [":nauseated_face:"],  # disgust
    8: [":astonished_face:"]  # surprised
}


def getEmoji(emotionID):
    return random.choice(emotion_map[emotionID])
