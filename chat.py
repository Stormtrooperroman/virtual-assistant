import random
import json
import torch
import os
import subprocess
import speech_recognition as sr
import pyttsx3
import queue
import sounddevice as sd
import vosk
import sys
import wikipedia
import webbrowser
import cv2
import face_recognition
import numpy as np
import requests
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from datetime import datetime, timedelta
from config import MAX_LENGTH, save_dir
from load import indexes_from_sentence, normalize_string, Voc
from pytgbot import Bot


q = queue.Queue()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\TokenEnums\RHVoice\Anna")
    rate = engine.getProperty("rate")
    engine.setProperty("rate", rate - 40)

except ImportError:
    print("Error 1")
    exit(1)
except RuntimeError:
    print("Error 2")
    exit(1)


speach = sr.Recognizer()


device_info = sd.query_devices(None, 'input')
samplerate = int(device_info['default_samplerate'])
model = vosk.Model("models/ru")
rec = vosk.KaldiRecognizer(model, samplerate)



roman_image = face_recognition.load_image_file("Roma.png")
roman_face_encoding = face_recognition.face_encodings(roman_image)[0]

known_face_encodings = [
    roman_face_encoding
]

known_face_names = [
    "Lider Roman"
]


def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def evaluate(searcher, vocabulary, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexes_from_sentence(vocabulary, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [vocabulary.index2word[token.item()] for token in tokens]
    return decoded_words


def speak(cmd):
    engine.say(cmd)
    engine.runAndWait()


def read_voice():

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16',
                           channels=1, callback=callback):
        while True:
            audio = q.get()
            if rec.AcceptWaveform(audio):
                voice_text = json.loads(rec.Result())["text"]
                print(voice_text)
                return voice_text


with open('intents.json', 'r', encoding="utf-8") as f:
    intents = json.load(f)
FILE = "data.pth"
data = torch.load(FILE)

FILE = "scripted_chatbot.pth"
scripted_searcher = torch.load(FILE)
scripted_searcher.to(device)
scripted_searcher.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
corpus_name = "rus corpus"
corpus = os.path.join("data", corpus_name)
datafile = os.path.join(corpus, "formatted_dialogues.txt")

model_name = 'cb_model'
attn_model = 'dot'
hidden_size_two = 500
encoder_n_layers = 2
decoder_n_layers = 2
checkpoint_iter = 4000
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size_two),
                            '{}_checkpoint.tar'.format(checkpoint_iter))
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
voc = Voc(corpus_name)
voc.__dict__ = checkpoint['voc_dict']

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "..."

wikipedia.set_lang("ru")

api_key = ""
weather_url = "https://api.openweathermap.org/data/2.5/weather?"

API_KEY = ''

bot = Bot(API_KEY)


def weather():
    speak("Где вы хотите узнать погоду?")
    location = read_voice()
    if location == "":
        speak("Я вас не услышала, сэр.")
        return
    url = weather_url + "appid=" + api_key + "&lang=ru&units=metric" + "&q=" + location
    json_data = requests.get(url).json()
    if json_data["cod"] != "404":
        coord = json_data["coord"]
        weather_data = json_data["main"]
        temp = weather_data["temp"]
        hum = weather_data["humidity"]
        desc = json_data["weather"][0]["description"]
        resp_string = f"Данные по локации {location}. Долгота {coord['lon']}. " \
                      f"Широта {coord['lat']}. Температура {temp} градусов Цельсия. " \
                      f"Влажность воздуха {hum} процентов. Описание погоды {desc}"
        print(resp_string)
        speak(resp_string)
    else:
        speak("Город не найден.")


def whoami():
    video_capture = cv2.VideoCapture(0)
    know_man = 0
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    for _ in range(200):
        ret, frame = video_capture.read()

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == "Unknown":
                color = (0, 0, 255)
            else:
                know_man += 1

                color = (0, 255, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        if know_man >= 20:
            video_capture.release()
            cv2.destroyAllWindows()
            return True
    video_capture.release()
    cv2.destroyAllWindows()

    return False


def new_year():
    date_now = datetime.today()
    date_now_timestamp = int(date_now.timestamp())
    new_year_date = datetime(year=date_now.year + 1, month=1, day=1, hour=0, minute=0, second=0)
    new_year_timestamp = int(new_year_date.timestamp())
    remaining_time_seconds = new_year_timestamp - date_now_timestamp

    remaining_time = timedelta(seconds=remaining_time_seconds)
    remaining_days = remaining_time.days
    remaining_seconds = remaining_time.seconds

    remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
    remaining_hours, remaining_minutes = divmod(remaining_minutes, 60)

    if remaining_days:
        if 1 < remaining_days % 10 < 5:
            remaining_days = f'{remaining_days} дня'
        elif remaining_days % 10 == 1 and remaining_days % 100 != 11:
            remaining_days = f'{remaining_days} день'
        else:
            remaining_days = f'{remaining_days} дней'
    else:
        remaining_days = ''

    if remaining_hours:
        if 1 < remaining_hours < 5:
            remaining_hours = f', {remaining_hours} часа'
        elif remaining_hours % 10 == 1 and remaining_hours != 11:
            remaining_hours = f', {remaining_hours} час'
        else:
            remaining_hours = f', {remaining_hours} часов'
    else:
        remaining_hours = ''

    if remaining_minutes:
        if 1 < remaining_minutes < 5:
            remaining_minutes = f', {remaining_minutes} минуты'
        elif remaining_minutes % 10 == 1 and remaining_minutes != 11:
            remaining_minutes = f', {remaining_minutes} минута'
        else:
            remaining_minutes = f', {remaining_minutes} минут'
    else:
        remaining_minutes = ''

    if remaining_seconds:
        if 1 < remaining_seconds < 5:
            remaining_seconds = f', {remaining_seconds} секунды'
        elif remaining_seconds % 10 == 1 and remaining_seconds != 11:
            remaining_seconds = f', {remaining_seconds} секунда'
        else:
            remaining_seconds = f', {remaining_seconds} секунд'
    else:
        remaining_seconds = ''
    speak(f'До Нового года осталось {remaining_days}{remaining_hours}{remaining_minutes}{remaining_seconds}')


def hello(response):
    clock = datetime.now().time()
    if 0 <= clock.hour < 12:
        a = random.choice([response[2], response[1]])
    elif 12 <= clock.hour < 18:
        a = random.choice([response[0], response[1]])
    else:
        a = random.choice([response[3], response[1]])
    speak(a)


def bye(response):
    speak(random.choice(response))
    exit()


def music():
    music_dir = 'C:/Users/lider/Desktop/Музыка'
    songs = os.listdir(music_dir)
    str_play = 'wmplayer'
    for song in songs:
        str_play += f' "{music_dir}/{song}"'
    subprocess.call(str_play)


def wiki():
    speak("Скажите то, что вы хотите найти в Википедии?")
    statement = read_voice()
    if statement == "":
        speak("Я вас не услышала, сэр.")
        return
    try:
        results = wikipedia.summary(statement, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        print(e.options)
        s = random.choice(e.options)
        results = wikipedia.summary(s)
    speak("В соответствии с Википедией")
    print(results)
    speak(results)


def browser():
    speak("Запуск браузера")
    webbrowser.open_new_tab("https://www.google.com")


def search():
    speak("Скажите то, что вы хотите найти?")
    statement = read_voice()
    if statement == "":
        speak("Я вас не услышала, сэр.")
        return
    webbrowser.open_new_tab(f"https://google.com/search?q={statement}")
    speak(f"Вот что найденно по  вашему запросу {statement}")


def place():
    speak("Скажите место, которое вы хотите увидеть на карте?")
    statement = read_voice()
    if statement == "":
        speak("Я вас не услышала, сэр.")
        return
    webbrowser.open_new_tab(f"https://google.com/maps/place/{statement}")
    speak(f"Карта по запросу {statement}")


def get_tag(text):
    sentence = tokenize(text)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return tag, prob


def send():
    while True:
        speak("Скажите то, что вы хотите отправить?")
        statement = read_voice()
        if statement == "":
            continue
        statement = f"{statement.capitalize()}."
        speak(f"Вы хотите отправить сообщение со следующим содержанием {statement}")

        answer = read_voice()
        tag, prob = get_tag(answer)
        if prob.item() > 0.75:

            if tag == "yes":
                break
            elif tag == "no":
                continue
            elif tag == "bye":
                speak("Остановка отправки сообщения")
                return
            else:
                speak("Ваш ответ не распознан. Повторите ещё")

    while True:
        speak("Скажите, куда вы хотите отправить сообщение?")
        chat = read_voice()
        if chat == "":
            continue
        tag, prob = get_tag(chat)
        chat_info = {}
        if prob.item() > 0.75:
            for intent in intents["chats"]:
                if tag == intent["tag"]:
                    chat_info = intent
                    break
            else:
                continue
        else:
            continue

        speak(f"Вы хотите отправить сообщение в {chat_info['response']} ?")

        answer = read_voice()
        tag, prob = get_tag(answer)
        if prob.item() > 0.75:

            if tag == "yes":
                break
            elif tag == "no":
                continue
            elif tag == "bye":
                speak("Остановка отправки сообщения")
                return
            else:
                speak("Ваш ответ не распознан. Повторите ещё")

    speak("Перед отправкой я должна подтвердить вашу личность,"
          "сэр. Пожалуйста посмотрите в камеру.")

    if not whoami():
        speak("Не удалось подтвердить личность")
        return
    speak("Ваша личность подтверждена, сэр. Отправка сообщения.")
    bot.send_message(chat_info["chat_id"], statement)
    speak(f"Ваше сообщение с содержанием: {statement}, отправленно")


def add_task():
    token = ''

    database_id = ''

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    create_url = 'https://api.notion.com/v1/pages'

    while True:
        speak("Скажите заголовок задание")
        title = read_voice()
        if title == "":
            continue
        title = f"{title.capitalize()}."
        speak(f"Вы хотите создать задание со следующим заголовком {title}")

        answer = read_voice()
        tag, prob = get_tag(answer)
        if prob.item() > 0.75:

            if tag == "yes":
                break
            elif tag == "no":
                continue
            elif tag == "bye":
                speak("Остановка создания задания")
                return
            else:
                speak("Ваш ответ не распознан. Повторите ещё")



    while True:
        speak("Скажите  задание")
        text = read_voice()
        if text == "":
            continue
        text = f"{text.capitalize()}."
        speak(f"Вы хотите создать следующие задание {text}")

        answer = read_voice()
        tag, prob = get_tag(answer)
        if prob.item() > 0.75:

            if tag == "yes":
                break
            elif tag == "no":
                continue
            elif tag == "bye":
                speak("Остановка создания задания")
                return
            else:
                speak("Ваш ответ не распознан. Повторите ещё")

    new_page_data = {
        "parent": {"database_id": database_id},
        "properties": {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            },
            "Task": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": text
                        }
                    }
                ]
            }
        }
    }

    task = json.dumps(new_page_data)
    res = requests.request("POST", create_url, headers=headers, data=task)
    print(res.status_code)
    speak("Задание успешно созданно")


def task_title(page_id, headers):
    data_url = f"https://api.notion.com/v1/pages/{page_id}/properties/title"
    res = requests.request("GET", data_url, headers=headers)
    data = res.json()
    return data["results"][0]["title"]["text"]["content"]


def read_task():
    token = ''
    database_id = ''

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    read_url = f"https://api.notion.com/v1/databases/{database_id}/query"

    filter_data = {
        "filter": {
            "property": "Checkbox",
            "checkbox": {
                "equals": False
            }
        }
    }
    filter_data = json.dumps(filter_data)
    res = requests.request("POST", read_url, headers=headers, data=filter_data)
    task_data = res.json()
    tasks = []
    for page in task_data["results"]:
        tasks.append(task_title(page["id"], headers))

    speak(f"У вас {len(tasks)} заданий")
    for i, task in enumerate(tasks):
        speak(f"Задание {i+1}: {task}")


def main():
    speak("Здравствуйте, сэр.")
    while True:
        sentence = read_voice()

        if sentence != "":
            input_sentence = sentence
            tag, prob = get_tag(sentence)
            if prob.item() > 0.75:
                for intent in intents["intents"]:
                    if tag == intent["tag"]:
                        out = ""

                        if intent["tag"] == "hello":
                            hello(intent["response"])

                        elif intent["tag"] == "bye":
                            bye(intent['response'])

                        elif intent["tag"] == "time":
                            clock = datetime.now().time()
                            out = random.choice(intent['response'])
                            out += str(clock.hour) + " " + str(clock.minute)
                            speak(out)

                        elif intent["tag"] == "newyear":
                            new_year()

                        elif intent["tag"] == "music":
                            music()

                        elif intent["tag"] == "wiki":
                            wiki()

                        elif intent["tag"] == "browser":
                            browser()

                        elif intent["tag"] == "search":
                            search()

                        elif intent["tag"] == "weather":
                            weather()

                        elif intent["tag"] == "place":
                            place()

                        elif intent["tag"] == "send":
                            send()

                        elif intent["tag"] == "add_task":
                            add_task()

                        elif intent["tag"] == "view_tasks":
                            read_task()

                        else:
                            a = random.choice(intent['response'])
                            text = a.split()
                            for i in text:
                                out = out + " " + i
                            speak(out)
                        break
            else:
                try:
                    input_sentence = normalize_string(input_sentence)
                    output_words = evaluate(scripted_searcher, voc, input_sentence)
                    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                    speak(' '.join(output_words))
                except KeyError:
                    speak("Я ещё учусь и пока вас не поняла.")
                    print(f"{bot_name}:Я ещё учусь и пока вас не поняла.")


if __name__ == '__main__':
    main()
