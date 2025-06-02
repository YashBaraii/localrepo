import os
import pyttsx3
import speech_recognition as sr
import webbrowser
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import threading
import queue
import time
import re
import sys
import signal
import atexit
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import json
import hashlib
import subprocess
import psutil
import pickle
import winsound
from pathlib import Path
import gc  # Garbage Collector
from contextlib import contextmanager

load_dotenv()

# Set up your API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

# Add these global variables
CONVERSATION_FILE = "conversation_history.json"
LEARNING_FILE = "learned_patterns.json"
MEMORY_FILE = "assistant_memory.pkl"
REMINDERS_FILE = "reminders.pkl"
conversation_history = []
learned_patterns = {}
memory_data = {}
reminders = []

active_timers = []
timer_threads = []

# Add this near the top of the file with other global variables
is_speaking = False
should_pause = False  # Changed from should_stop to should_pause
speech_queue = queue.Queue()
engine = None  # Global engine instance
engine_lock = threading.Lock()
current_speech_thread = None

# Add these global variables
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # Add this to your .env file
DEFAULT_CITY = "Nagpur"  # Default city

CONVERSATIONS_DIR = "conversations"
current_conversation_id = None

# Add these global variables
OUTPUT_DIR = "./generated_files/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Add these global variables
is_listening = False
last_activity_time = None
TIMEOUT_DURATION = 30  # seconds to wait before going to sleep
WAKE_WORDS = ["hey max", "hi max", "hello max", "wake up max"]

# Add these global variables
MAX_SPEECH_INSTANCES = 3  # Limit concurrent speech instances
SPEECH_MEMORY_LIMIT = 1024 * 1024 * 100  # 100MB limit for speech engine


def format_code_response(response):
    """Format code responses with proper markdown and syntax."""
    # Check if response contains code
    if "```" not in response:
        # If it looks like code but not formatted, wrap it
        if any(
            keyword in response.lower()
            for keyword in ["def ", "print(", "return ", "import "]
        ):
            lines = response.split("\n")
            formatted = "Here's the code:\n```python\n"
            for line in lines:
                formatted += line.strip() + "\n"
            formatted += "```\n\nOutput example:\n```\n"
            # Add example output if available
            if "Output:" in response:
                output = response.split("Output:")[1].strip()
                formatted += output + "\n"
            formatted += "```"
            return formatted
    return response


def chat_with_gemini(user_input):
    """Chat with Gemini model."""
    global current_conversation_id, last_activity_time

    try:
        # Create a system prompt that defines Max's identity
        system_prompt = """You are Max, a virtual personal assistant created for Sir Yash. 
        Always identify yourself as Max and maintain this identity consistently. 
        Never say you are Gemini or any other AI model. 
        You are helpful, polite, and dedicated to assisting Sir Yash with various tasks."""

        # Combine system prompt with user input
        full_prompt = f"{system_prompt}\n\nUser: {user_input}\nMax:"

        # Generate response from Gemini
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)

        # Clean and format the response
        cleaned_response = clean_response(response.text)

        # Ensure the response aligns with Max's identity
        if "gemini" in cleaned_response.lower() or "google" in cleaned_response.lower():
            cleaned_response = "Hello! I'm Max, your virtual personal assistant. How can I help you today, Sir Yash?"

        # Always speak the response
        speak(cleaned_response)

        return cleaned_response

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        speak(error_msg)
        return error_msg


def answer_question(user_input, is_joke=False):
    """Answer questions using Gemini model with proper formatting."""
    try:
        if is_joke:
            prompt = "Tell me a short, clean, and funny joke."
        else:
            prompt = user_input

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        # Format the response if it contains code
        formatted_response = format_code_response(response.text)

        # Speak the response without code formatting
        speak_text = formatted_response.replace("```python", "").replace("```", "")
        speak(speak_text)

        return formatted_response

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        speak(error_msg)
        return error_msg


def search_and_speak(query):
    """Open browser with search query for Google or YouTube."""
    # Extract the actual search query by removing search keywords
    search_keywords = [
        "search",
        "look up",
        "find",
        "google",
        "search for",
        "search about",
    ]
    youtube_keywords = ["youtube", "video", "videos", "watch"]
    query_words = query.lower().split()

    # Check if this is a YouTube search
    is_youtube_search = any(keyword in query.lower() for keyword in youtube_keywords)

    # Remove search keywords from the beginning of the query
    while query_words and any(
        query_words[0] in keyword for keyword in (search_keywords + youtube_keywords)
    ):
        query_words.pop(0)

    # Reconstruct the cleaned query
    cleaned_query = " ".join(query_words)

    if not cleaned_query:
        speak("Please specify what you'd like me to search for.")
        return

    if is_youtube_search:
        display_message("Assistant", f"Searching YouTube for: {cleaned_query}")
        speak(f"Opening YouTube to search for {cleaned_query}")
        search_url = (
            f"https://www.youtube.com/results?search_query={quote(cleaned_query)}"
        )
    else:
        display_message("Assistant", f"Searching Google for: {cleaned_query}")
        speak(f"Opening browser to search for {cleaned_query}")
        search_url = f"https://www.google.com/search?q={quote(cleaned_query)}"

    # Open browser with the search query
    webbrowser.open(search_url)


def get_weather(location=None):
    """Get weather information for a location using OpenWeatherMap API."""
    if not OPENWEATHER_API_KEY:
        print("\033[91mWeather API key is not configured in .env file\033[0m")
        speak(
            "Weather API key is not configured. Please set up the OpenWeatherMap API key."
        )
        return

    try:
        if not location or location.strip() == "":
            location = DEFAULT_CITY

        print(f"\033[94mFetching weather for: {location}\033[0m")

        # Get coordinates first (geocoding)
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_response = requests.get(geo_url, timeout=10)

        if geo_response.status_code == 401:
            print(
                "\033[91mInvalid API key. Please check your OpenWeatherMap API key in .env file\033[0m"
            )
            speak(
                "The weather service API key is invalid. Please check your API key configuration."
            )
            return

        if geo_response.status_code != 200:
            print(
                f"\033[91mGeocoding API error. Status code: {geo_response.status_code}\033[0m"
            )
            speak(
                f"Sorry, I couldn't access the weather service. Status code: {geo_response.status_code}"
            )
            return

        geo_data = geo_response.json()

        if not geo_data:
            speak(f"Sorry, I couldn't find the location: {location}")
            return

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # Get weather data
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        weather_response = requests.get(weather_url, timeout=10)

        if weather_response.status_code != 200:
            print(
                f"\033[91mWeather API error. Status code: {weather_response.status_code}\033[0m"
            )
            speak("Sorry, I couldn't fetch the weather information at the moment.")
            return

        weather_data = weather_response.json()

        # Extract weather information
        temp = round(float(weather_data["main"]["temp"]))
        feels_like = round(float(weather_data["main"]["feels_like"]))
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data["wind"]["speed"]
        description = weather_data["weather"][0]["description"]
        city_name = weather_data["name"]

        # Create weather message
        weather_msg = f"Current weather in {city_name}:\n"
        weather_msg += f"Temperature is {temp}°C, feels like {feels_like}°C\n"
        weather_msg += f"Conditions: {description.capitalize()}\n"
        weather_msg += f"Humidity: {humidity}%\n"
        weather_msg += f"Wind speed: {wind_speed} meters per second"

        print(f"\033[94mWeather data fetched successfully\033[0m")
        speak(weather_msg)
        return weather_msg

    except requests.exceptions.Timeout:
        speak(
            "Sorry, the weather service is taking too long to respond. Please try again."
        )
        return
    except requests.exceptions.RequestException as e:
        print(f"\033[91mNetwork error: {str(e)}\033[0m")
        speak(f"Sorry, there was a network error: {str(e)}")
        return
    except Exception as e:
        print(f"\033[91mUnexpected error: {str(e)}\033[0m")
        speak(f"Sorry, there was an unexpected error: {str(e)}")
        return


def extract_location(user_input):
    """Extract location from user input."""
    # Common weather-related phrases to remove
    weather_phrases = [
        "what's the weather in",
        "what is the weather in",
        "what's the weather of",
        "what is the weather of",
        "weather in",
        "weather at",
        "weather for",
        "weather of",
        "temperature in",
        "temperature at",
        "temperature for",
        "temperature of",
        "how's the weather in",
        "how is the weather in",
        "get weather",
        "check weather",
        "what's the weather",
        "how's the weather",
        "tell me the weather in",
    ]

    # Remove weather phrases
    cleaned_input = user_input.lower()
    for phrase in weather_phrases:
        if cleaned_input.startswith(phrase):
            cleaned_input = cleaned_input.replace(phrase, "", 1)
            break
        elif phrase in cleaned_input:
            cleaned_input = cleaned_input.split(phrase, 1)[1]
            break

    # Clean up and return location
    location = cleaned_input.strip()
    print(f"\033[94mExtracted location: {location}\033[0m")  # Debug print
    return location if location else DEFAULT_CITY


def open_application(app_name):
    """Open specified application or website."""
    app_commands = {
        "notepad": "notepad",
        "visual studio code": "code",
        "terminal": "cmd",
        "word": "start winword",
        "excel": "start excel",
        "powerpoint": "start powerpnt",
        "settings": "start ms-settings:",
        "camera": "start microsoft.windows.camera:",
        "paint": "mspaint",
        "spotify": "start spotify",
        "calculator": "calc",
        "task manager": "taskmgr",
        "file explorer": "explorer",
        "control panel": "control",
        "powershell": "powershell",
    }

    web_links = {
        "youtube": "https://www.youtube.com/",
        "google": "https://www.google.com/",
        "github": "https://github.com/YashBaraii",
        "pinterest": "https://in.pinterest.com/",
        "speed test": "https://fast.com/",
        "chat gpt": "https://chatgpt.com/",
        "portfolio": "https://yashbarai.netlify.app/",
        "linkedin": "https://www.linkedin.com/in/yash-baraii/",
        "twitter": "https://x.com/yashbaraii",
        "instagram": "https://www.instagram.com/yashbaraii/",
        "flipkart": "https://www.flipkart.com/",
        "amazon": "https://www.amazon.in/",
        "whatsapp": "https://web.whatsapp.com/",
        "canva": "https://www.canva.com/",
    }

    try:
        app_name = app_name.lower()

        # Check if it's a web link
        if app_name in web_links:
            webbrowser.open(web_links[app_name])
            response = f"Opening {app_name.title()} in your browser, Sir."
        # Check if it's an application
        elif app_name in app_commands:
            subprocess.Popen(app_commands[app_name], shell=True)
            response = f"Opening {app_name.title()}, Sir."
        else:
            response = f"Sorry, I don't know how to open {app_name}. Please try another application or website."

        speak(response)
        return response

    except Exception as e:
        error_msg = f"Sorry, I couldn't open {app_name}: {str(e)}"
        speak(error_msg)
        return error_msg


def close_application(app_name):
    """Close specified application or web link."""
    app_processes = {
        "notepad": "notepad.exe",
        "vs code": "Code.exe",
        "word": "WINWORD.EXE",
        "excel": "EXCEL.EXE",
        "powerpoint": "POWERPNT.EXE",
        "settings": "SystemSettings.exe",
        "camera": "WindowsCamera.exe",
        "calculator": "Calculator.exe",
        "paint": "mspaint.exe",
        "task manager": "Taskmgr.exe",
        "file explorer": "explorer.exe",
        "spotify": "Spotify.exe",
        "terminal": "cmd.exe",
        "powershell": "powershell.exe",
        "chrome": "chrome.exe",
        "firefox": "firefox.exe",
        "edge": "msedge.exe",
        "brave": "brave.exe",
    }

    web_links = {
        "youtube": "youtube.com",
        "google": "google.com",
        "github": "github.com",
        "pinterest": "pinterest.com",
        "speed test": "fast.com",
        "chat gpt": "chat.openai.com",
        "portfolio": "yashbarai.netlify.app",
        "linkedin": "https://www.linkedin.com/in/yash-baraii/",
        "twitter": "https://x.com/yashbaraii",
        "instagram": "https://www.instagram.com/yashbaraii/",
        "flipkart": "https://www.flipkart.com/",
        "amazon": "https://www.amazon.in/",
        "whatsapp": "https://web.whatsapp.com/",
        "canva": "https://www.canva.com/",
    }

    try:
        app_name = app_name.lower()
        closed = False

        # First check if it's a web link
        if app_name in web_links:
            domain = web_links[app_name]
            # Try closing in different browsers
            browsers = ["chrome.exe", "firefox.exe", "msedge.exe", "brave.exe"]
            for browser in browsers:
                try:
                    # PowerShell command to close specific tabs
                    ps_command = f"""
                    $browser = Get-Process "{browser.split('.')[0]}" -ErrorAction SilentlyContinue
                    if ($browser) {{
                        $shell = New-Object -ComObject Shell.Application
                        $windows = $shell.Windows()
                        $windows | Where-Object {{$_.LocationURL -like "*{domain}*"}} | ForEach-Object {{$_.Quit()}}
                    }}
                    """
                    subprocess.run(
                        ["powershell", "-Command", ps_command], capture_output=True
                    )
                    closed = True
                except Exception as e:
                    print(f"\033[91mError closing {browser}: {e}\033[0m")

        # Then check if it's an application
        if app_name in app_processes:
            process_name = app_processes[app_name]
            for proc in psutil.process_iter(["name"]):
                if proc.info["name"].lower() == process_name.lower():
                    proc.kill()
                    closed = True

        if closed:
            response = f"Closed {app_name.title()}, Sir."
        else:
            response = f"Couldn't find {app_name.title()} running."

        speak(response)
        return response

    except Exception as e:
        error_msg = f"Sorry, I couldn't close {app_name}: {str(e)}"
        speak(error_msg)
        return error_msg


def get_function_from_gemini(user_input):
    """Identifies which function to call and extracts parameters if needed."""
    user_input_lower = user_input.lower()

    # Check for file creation requests first
    file_keywords = [
        "in separate file",
        "in a file",
        "to a file",
        "save as",
        "save to file",
        "create file",
        "write to file",
        "generate file",
        "name it",
    ]

    if any(keyword in user_input_lower for keyword in file_keywords):
        print("\033[94mDetected file creation request\033[0m")
        return "generate_file", user_input, None

    # Memory recall (check first with specific phrase)
    if user_input_lower.startswith("do you remember"):
        return (
            "recall",
            user_input_lower.replace("do you remember", "", 1).strip(),
            None,
        )

    # Remember command
    if user_input_lower.startswith("remember that"):
        text = user_input_lower.replace("remember that", "", 1).strip()
        if " is " in text:
            key, value = text.split(" is ", 1)
            return "remember", key.strip(), value.strip()

    # Weather related queries
    weather_patterns = [
        "what's the weather",
        "what is the weather",
        "how's the weather",
        "how is the weather",
        "weather in",
        "weather at",
        "weather of",
        "temperature in",
        "temperature at",
        "temperature of",
    ]

    if any(pattern in user_input_lower for pattern in weather_patterns):
        location = extract_location(user_input)
        print(f"\033[94mDetected weather query for location: {location}\033[0m")
        return "get_weather", location, None

    # Timer commands
    if any(word in user_input_lower for word in ["set timer", "start timer"]):
        duration = user_input_lower.split("timer", 1)[1].strip()
        return "set_timer", duration, None

    # Application and website control commands
    open_keywords = ["open", "launch", "start", "run"]
    close_keywords = ["close", "exit", "quit", "terminate"]

    # Check for open commands
    if any(user_input_lower.startswith(keyword + " ") for keyword in open_keywords):
        for keyword in open_keywords:
            if user_input_lower.startswith(keyword + " "):
                app_name = user_input_lower.replace(keyword, "", 1).strip()
                return "open_application", app_name, None

    # Check for close commands
    if any(user_input_lower.startswith(keyword + " ") for keyword in close_keywords):
        for keyword in close_keywords:
            if user_input_lower.startswith(keyword + " "):
                app_name = user_input_lower.replace(keyword, "", 1).strip()
                return "close_application", app_name, None

    # Search related queries
    search_keywords = ["search", "look up", "find", "google", "search for"]
    if any(keyword in user_input_lower for keyword in search_keywords):
        return "search_and_speak", user_input, None

    # Time related queries
    time_keywords = ["what time", "current time", "tell me the time", "what's the time"]
    if any(keyword in user_input_lower for keyword in time_keywords):
        return "get_current_time", None, None

    # Default to chat
    return "chat_with_gemini", user_input, None


def clean_response(text):
    """Clean the response text by removing asterisks and unnecessary characters."""
    # Remove content within asterisks
    text = re.sub(r"\*.*?\*", "", text)
    # Remove any remaining asterisks
    text = text.replace("*", "")
    # Remove multiple spaces
    text = " ".join(text.split())
    # Remove any markdown formatting
    text = re.sub(r"[#*_`~]", "", text)
    # Remove any URLs
    text = re.sub(r"http\S+|www.\S+", "", text)
    # Normalize punctuation
    text = text.replace("...", ".")
    text = text.replace("..", ".")
    text = text.replace(",.", ".")
    return text.strip()


def initialize_engine():
    """Initialize the text-to-speech engine with memory optimization."""
    global engine
    try:
        # Force garbage collection before creating new engine
        gc.collect()

        if engine is not None:
            try:
                engine.stop()
                del engine
            except:
                pass

        # Set process memory priority
        try:
            import psutil

            process = psutil.Process()
            process.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows
        except:
            pass

        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        return engine
    except:
        return pyttsx3.init()


def display_message(speaker, message):
    """Display formatted conversation messages."""
    if speaker.lower() == "assistant":
        print(f"\033[94m{speaker}: {message}\033[0m")  # Blue color for assistant
    else:
        print(f"\033[92m{speaker}: {message}\033[0m")  # Green color for user


def initialize_speech_engine():
    """Initialize the text-to-speech engine."""
    global engine
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        return True
    except Exception as e:
        print(f"Error initializing speech engine: {e}")
        return False


def speak(text):
    """Speak the given text."""
    global engine, is_speaking, should_pause, current_speech_thread

    try:
        with engine_lock:  # Use lock for thread safety
            # Initialize engine if not exists
            if not engine:
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.setProperty("volume", 1.0)

            is_speaking = True
            should_pause = False

            def speak_text():
                try:
                    if not should_pause:
                        engine.say(text)
                        engine.runAndWait()
                        engine.endLoop()  # End the loop after speaking
                except Exception as e:
                    print(f"Error in speech thread: {e}")
                finally:
                    global is_speaking
                    is_speaking = False

            # Stop existing thread if running
            if current_speech_thread and current_speech_thread.is_alive():
                should_pause = True
                current_speech_thread.join(timeout=0.5)

            # Create and start new speech thread
            should_pause = False
            current_speech_thread = threading.Thread(target=speak_text)
            current_speech_thread.daemon = True
            current_speech_thread.start()

    except Exception as e:
        print(f"Error initializing speech: {e}")
        is_speaking = False


def listen():
    """Listen for user input with ability to pause speaking."""
    global is_speaking, should_pause, current_speech_thread, is_listening, last_activity_time
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=0.5)

            if is_listening:
                print("\nListening...")
            else:
                print("\nSleeping... (Say 'Hey Max' to wake me up)")

            try:
                # Shorter timeout when sleeping to be more responsive to wake word
                timeout = 5 if not is_listening else 10
                audio = r.listen(source, timeout=timeout, phrase_time_limit=5)

                try:
                    user_input = r.recognize_google(audio).lower()

                    # Check for sleep commands when listening
                    if is_listening and any(
                        phrase in user_input
                        for phrase in [
                            "you can rest",
                            "you can rest now",
                            "go to sleep",
                        ]
                    ):
                        is_listening = False
                        speak("Thank you, I'll be here if you need me.")
                        return None

                    # Check if this is a wake word when not listening
                    if not is_listening:
                        if any(wake_word in user_input for wake_word in WAKE_WORDS):
                            print(f"\033[94mWake word detected: {user_input}\033[0m")
                            is_listening = True
                            last_activity_time = time.time()
                            speak("Yes Sir?")
                            return None
                        return None

                    display_message("User", user_input)
                    last_activity_time = time.time()

                    # Handle stop command with more variations
                    if is_speaking and any(
                        word in user_input
                        for word in ["stop", "pause", "quiet", "shut up", "be quiet"]
                    ):
                        should_pause = True
                        if current_speech_thread and current_speech_thread.is_alive():
                            current_speech_thread.join(timeout=0.5)
                        print("\033[93mSpeech stopped.\033[0m")
                        return None

                    # Handle exit commands
                    if any(
                        word in user_input
                        for word in ["exit", "quit", "goodbye", "bye"]
                    ):
                        return "exit"

                    return user_input

                except sr.UnknownValueError:
                    if is_listening:
                        print("\033[91mCould not understand audio\033[0m")
                    return None

            except sr.WaitTimeoutError:
                return None

    except KeyboardInterrupt:
        print("\n\033[91mKeyboard interrupt detected.\033[0m")
        return "exit"
    except Exception as e:
        print(f"\n\033[91mCritical error: {str(e)}\033[0m")
        return "exit"


def get_current_time():
    """Get and speak the current time."""
    current_time = datetime.now().strftime("%I:%M %p")
    response = f"The current time is {current_time}"
    speak(response)
    return response


def open_close_tasks(action, app_name):
    """Open or close applications based on the given action and app name."""
    app_commands = {
        "notepad": "notepad",
        "visual studio code": "code",
        "terminal": "cmd",
        "word": "start winword",
        "excel": "start excel",
        "powerpoint": "start powerpnt",
        "settings": "start ms-settings:",
        "camera": "start microsoft.windows.camera:",
        "paint": "mspaint",
        "spotify": "start spotify",
    }
    app_processes = {
        "notepad": "notepad.exe",
        "vs code": "Code.exe",
        "word": "WINWORD.EXE",
        "excel": "EXCEL.EXE",
        "powerpoint": "POWERPNT.EXE",
        "settings": "SystemSettings.exe",
        "camera": "WindowsCamera.exe",
    }
    web_links = {
        "youtube": "https://www.youtube.com/",
        "google": "https://www.google.com/",
        "github": "https://github.com/YashBaraii",
        "pinterest": "https://in.pinterest.com/",
        "speed test": "https://fast.com/",
        "chat gpt": "https://chatgpt.com/",
        "portfolio": "https://yashbarai.netlify.app/",
    }
    app_name = app_name.lower()

    if action == "open":
        if app_name in app_commands:
            speak(f"Opening {app_name}.")
            os.system(app_commands[app_name])
            return
        if app_name in web_links:
            speak(f"Opening {app_name}.")
            webbrowser.open(web_links[app_name])
            return

    elif action == "close":
        if app_name in app_processes:
            speak(f"Closing {app_name}.")
            os.system(f"taskkill /f /im {app_processes[app_name]}")
            return

    speak("I couldn't find the requested application or website.")


def list_conversations():
    """List all saved conversations."""
    try:
        conversations = []
        for filename in os.listdir(CONVERSATIONS_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(CONVERSATIONS_DIR, filename)
                with open(file_path, "r") as f:
                    conversation = json.load(f)
                    first_message = conversation["messages"][0]["user_input"]
                    conversations.append(
                        {
                            "id": conversation["conversation_id"],
                            "start_time": conversation["start_time"],
                            "first_message": first_message,
                        }
                    )

        if conversations:
            response = "Here are your saved conversations:\n\n"
            for conv in conversations:
                response += f"ID: {conv['id']}\n"
                response += f"Started: {conv['start_time']}\n"
                response += f"First message: {conv['first_message']}\n\n"
        else:
            response = "No saved conversations found."

        speak(response)
        return response

    except Exception as e:
        error_msg = f"Error listing conversations: {str(e)}"
        speak(error_msg)
        return error_msg


def load_conversation(conversation_id):
    """Load and display a specific conversation."""
    try:
        file_path = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                conversation = json.load(f)

            response = f"Conversation {conversation_id}:\n\n"
            for message in conversation["messages"]:
                response += f"[{message['timestamp']}]\n"
                response += f"User: {message['user_input']}\n"
                response += f"Assistant: {message['assistant_response']}\n\n"

            speak(response)
            return response
        else:
            error_msg = f"Conversation {conversation_id} not found."
            speak(error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"Error loading conversation: {str(e)}"
        speak(error_msg)
        return error_msg


def start_new_conversation():
    """Start a new conversation."""
    global current_conversation_id
    current_conversation_id = None
    speak("Starting a new conversation.")


# Add these functions before the function_map definition
def remember(key, value):
    """Store information in memory."""
    # Clean up the key by removing common prefixes
    key = key.lower().replace("my ", "").replace("that ", "").strip()
    memory_data[key] = value
    save_memory()
    response = f"I'll remember that {key} is {value}"
    speak(response)
    return response


def recall(user_input):
    """Recall information from memory."""
    # Clean up the query
    key = user_input.lower().strip()

    # Remove common question prefixes
    prefixes = ["what's my ", "what is my ", "what's ", "what is ", "about ", "my "]
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key.replace(prefix, "", 1)

    if key in memory_data:
        response = f"Yes, I remember that {key} is {memory_data[key]}"
    else:
        response = f"No, I don't have any memory about {key}"
    speak(response)
    return response


def set_timer(duration_str):
    """Set a timer for specified duration."""
    try:
        duration = parse_duration_string(duration_str)
        if duration <= 0:
            response = "Please specify a valid duration."
            speak(response)
            return response

        end_time = datetime.now() + timedelta(seconds=duration)
        timer = {"end_time": end_time, "duration": duration_str}
        active_timers.append(timer)

        def timer_thread():
            time.sleep(duration)
            if timer in active_timers:
                active_timers.remove(timer)
                winsound.Beep(1000, 1000)
                response = f"Timer for {duration_str} is done!"
                speak(response)

        thread = threading.Thread(target=timer_thread)
        thread.daemon = True
        thread.start()
        timer_threads.append(thread)

        response = f"Timer set for {duration_str}"
        speak(response)
        return response

    except Exception as e:
        response = f"Sorry, I couldn't set that timer: {str(e)}"
        speak(response)
        return response


def set_reminder(text, time_str):
    """Set a reminder for a specific time."""
    try:
        reminder_time = parse_time_string(time_str)

        if reminder_time <= datetime.now():
            response = "Sorry, I can't set reminders for the past."
            speak(response)
            return response

        reminder = {
            "text": text,
            "time": reminder_time.strftime("%Y-%m-%d %H:%M"),
            "completed": False,
        }

        reminders.append(reminder)
        save_reminders()

        schedule_reminder(reminder)

        response = f"I'll remind you to {text} at {reminder_time.strftime('%I:%M %p on %B %d')}"
        speak(response)
        return response

    except Exception as e:
        response = f"Sorry, I couldn't set that reminder: {str(e)}"
        speak(response)
        return response


def save_to_file(content, file_type="text", custom_filename=None):
    """Save content to a file with appropriate naming."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if custom_filename:
        # Clean the filename of invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', "", custom_filename)
        filename = filename.replace(" ", "_")
    else:
        filename = f"{file_type}_{timestamp}"

    # Add extension if not present
    if not filename.endswith(".txt"):
        filename += ".txt"

    file_path = os.path.join(OUTPUT_DIR, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path
    except Exception as e:
        print(f"\033[91mError saving file: {e}\033[0m")
        return None


def generate_file(user_input):
    """Generate content and save to file based on user request."""
    try:
        # Ensure output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"\033[94mCreated output directory: {OUTPUT_DIR}\033[0m")

        # Extract the main request without the file keywords
        file_keywords = [
            "in separate file",
            "in a file",
            "to a file",
            "save as",
            "save to file",
            "create file",
            "write to file",
            "generate file",
        ]

        main_request = user_input
        file_type = "document"
        custom_filename = None

        # Extract filename if specified
        if "name it" in user_input.lower():
            parts = user_input.split("name it")
            if len(parts) > 1:
                custom_filename = parts[1].strip()
                main_request = parts[0]
                print(f"\033[94mCustom filename requested: {custom_filename}\033[0m")

        # Remove file-related keywords from the main request
        for keyword in file_keywords:
            if keyword.lower() in main_request.lower():
                main_request = main_request.replace(keyword, "").strip()

        print(f"\033[94mProcessing request: {main_request}\033[0m")

        # Determine content type and set appropriate prompt
        if "letter" in main_request.lower():
            file_type = "letter"
            prompt = f"Write a professional {main_request}"
        elif "code" in main_request.lower() or "program" in main_request.lower():
            file_type = "code"
            prompt = f"Write {main_request} with comments explaining the code"
        elif "report" in main_request.lower():
            file_type = "report"
            prompt = f"Generate a detailed {main_request}"
        else:
            prompt = main_request

        print(f"\033[94mContent type detected: {file_type}\033[0m")

        # Generate content using Gemini
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        content = response.text

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_filename:
            # Clean the filename
            filename = re.sub(r'[<>:"/\\|?*]', "", custom_filename)
            filename = filename.replace(" ", "_")
        else:
            filename = f"{file_type}_{timestamp}"

        # Add extension if not present
        if not filename.endswith(".txt"):
            filename += ".txt"

        # Create full file path using absolute path
        file_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
        print(f"\033[94mAttempting to save file: {file_path}\033[0m")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save content to file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"\033[92mFile saved successfully: {file_path}\033[0m")

            response = f"I've created the file at:\n{file_path}\n\nHere's what I generated:\n\n{content}"
            speak(f"I've generated the content and saved it to {filename}")

        except Exception as e:
            print(f"\033[91mError writing file: {e}\033[0m")
            response = f"Sorry, I couldn't save the file due to: {str(e)}\nBut here's the content:\n\n{content}"
            speak("I couldn't save the file, but I can show you the content.")

        return response

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(f"\033[91mError in generate_file: {e}\033[0m")
        speak(error_msg)
        return error_msg


def handle_command(command):
    """Handle system commands like open, close, search."""
    command = command.lower().strip()

    try:
        # Handle open commands
        if command.startswith("open "):
            app_name = command[5:].strip()
            return handle_open_command(app_name)

        # Handle close commands
        elif command.startswith("close "):
            app_name = command[6:].strip()
            return handle_close_command(app_name)

        # Handle search commands
        elif command.startswith("search "):
            search_query = command[7:].strip()
            return handle_search_command(search_query)

        else:
            return "I don't understand that command."

    except Exception as e:
        print(f"\033[91mError in handle_command: {str(e)}\033[0m")
        return f"Error executing command: {str(e)}"


def handle_open_command(app_name):
    """Handle opening applications or websites."""
    common_apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "chrome": "chrome.exe",
        "youtube": "https://youtube.com",
        "google": "https://google.com",
        "github": "https://github.com",
        "chat gpt": "https://chat.openai.com",
        "portfolio": "https://your-portfolio-url.com",  # Replace with your portfolio URL
        "linkedin": "https://linkedin.com",
    }

    try:
        if app_name in common_apps:
            target = common_apps[app_name]
            if target.startswith("http"):
                webbrowser.open(target)
            else:
                subprocess.Popen(target)
            response = f"Opening {app_name.title()}, Sir."
            speak(response)
            return response
        else:
            response = f"Sorry, I don't know how to open {app_name}. Please try another application or website."
            speak(response)
            return response
    except Exception as e:
        error_msg = f"Error opening {app_name}: {str(e)}"
        speak(error_msg)
        return error_msg


def handle_close_command(app_name):
    """Handle closing applications."""
    common_apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "chrome": "chrome.exe",
    }

    try:
        if app_name in common_apps:
            process_name = common_apps[app_name]
            for proc in psutil.process_iter():
                try:
                    if proc.name().lower() == process_name.lower():
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            response = f"Closed {app_name.title()}, Sir."
            speak(response)
            return response
        else:
            response = f"Sorry, I don't know how to close {app_name}."
            speak(response)
            return response
    except Exception as e:
        error_msg = f"Error closing {app_name}: {str(e)}"
        speak(error_msg)
        return error_msg


def handle_search_command(query):
    """Handle search queries."""
    try:
        if query.startswith(("youtube ", "yt ")):
            # Extract the actual search term
            search_term = query.replace("youtube ", "").replace("yt ", "")
            url = f"https://www.youtube.com/results?search_query={quote(search_term)}"
            webbrowser.open(url)
        else:
            url = f"https://www.google.com/search?q={quote(query)}"
            webbrowser.open(url)
        return None  # Return None as the response will be handled by the browser
    except Exception as e:
        return f"Error performing search: {str(e)}"


# Now define the function_map
function_map = {
    "get_current_time": get_current_time,
    "open_close_tasks": open_close_tasks,
    "chat_with_gemini": chat_with_gemini,
    "answer_question": answer_question,
    "search_and_speak": search_and_speak,
    "get_weather": get_weather,
    "list_conversations": list_conversations,
    "load_conversation": load_conversation,
    "start_new_conversation": start_new_conversation,
    "open_application": open_application,
    "close_application": close_application,
    "remember": remember,
    "recall": recall,
    "set_timer": set_timer,
    "set_reminder": set_reminder,
    "generate_file": generate_file,
    "handle_command": handle_command,
}


def load_learned_patterns():
    """Load learned patterns from file."""
    global learned_patterns
    try:
        if os.path.exists(LEARNING_FILE):
            with open(LEARNING_FILE, "r") as f:
                learned_patterns = json.load(f)
    except Exception as e:
        print(f"\033[91mError loading learned patterns: {e}\033[0m")
        learned_patterns = {}


def save_learned_patterns():
    """Save learned patterns to file."""
    try:
        with open(LEARNING_FILE, "w") as f:
            json.dump(learned_patterns, f, indent=2)
    except Exception as e:
        print(f"\033[91mError saving learned patterns: {e}\033[0m")


def generate_conversation_id(first_message):
    """Generate a unique conversation ID based on first message."""
    # Get the first few words (up to 3) of the first message
    words = first_message.split()[:3]
    initials = "".join(word[0].upper() for word in words)

    # Add timestamp to make it unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combine initials and timestamp
    conversation_id = f"{initials}_{timestamp}"

    return conversation_id


def initialize_conversation_storage():
    """Initialize the conversations directory."""
    if not os.path.exists(CONVERSATIONS_DIR):
        os.makedirs(CONVERSATIONS_DIR)
        print(f"\033[94mCreated conversations directory: {CONVERSATIONS_DIR}\033[0m")


def save_conversation(user_input, assistant_response, function_used):
    """Save conversation details to a unique file."""
    global current_conversation_id, conversation_history

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate conversation ID for new conversations
    if current_conversation_id is None:
        current_conversation_id = generate_conversation_id(user_input)
        print(f"\033[94mNew conversation started: {current_conversation_id}\033[0m")

    conversation_entry = {
        "timestamp": timestamp,
        "user_input": user_input,
        "assistant_response": assistant_response,
        "function_used": function_used,
    }

    try:
        # Create file path for this conversation
        file_path = os.path.join(CONVERSATIONS_DIR, f"{current_conversation_id}.json")

        # Load existing conversation or create new
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                conversation = json.load(f)
        else:
            conversation = {
                "conversation_id": current_conversation_id,
                "start_time": timestamp,
                "messages": [],
            }

        # Add new message
        conversation["messages"].append(conversation_entry)
        conversation["last_updated"] = timestamp

        # Save conversation
        with open(file_path, "w") as f:
            json.dump(conversation, f, indent=2)

        # Learn from this interaction
        learn_from_interaction(user_input, function_used)

    except Exception as e:
        print(f"\033[91mError saving conversation: {e}\033[0m")


def learn_from_interaction(user_input, function_used):
    """Learn patterns from user interactions."""
    global learned_patterns

    # Extract key words from user input
    words = user_input.lower().split()

    # Update learned patterns
    for word in words:
        if word not in learned_patterns:
            learned_patterns[word] = {}

        if function_used not in learned_patterns[word]:
            learned_patterns[word][function_used] = 1
        else:
            learned_patterns[word][function_used] += 1

    # Save updated patterns
    save_learned_patterns()


def get_suggested_function(user_input):
    """Get suggested function based on learned patterns."""
    words = user_input.lower().split()
    function_scores = {}

    for word in words:
        if word in learned_patterns:
            for func, count in learned_patterns[word].items():
                if func not in function_scores:
                    function_scores[func] = 0
                function_scores[func] += count

    if function_scores:
        return max(function_scores.items(), key=lambda x: x[1])[0]
    return None


def log_conversation(timestamp, user_input, response, function_used="chat_with_gemini"):
    """Log conversation to a JSON file with a unique ID."""
    try:
        # Create a unique conversation ID using timestamp
        date_str = datetime.now().strftime("%Y%m%d")
        conversation_id = f"H_{date_str}_{datetime.now().strftime('%H%M%S')}"

        # Create conversations directory if it doesn't exist
        if not os.path.exists("conversations"):
            os.makedirs("conversations")

        # Create or load conversation file
        filename = f"conversations/{conversation_id}.json"

        if os.path.exists(filename):
            with open(filename, "r") as f:
                conversation = json.load(f)
        else:
            conversation = {
                "conversation_id": conversation_id,
                "start_time": timestamp,
                "messages": [],
            }

        # Add new message
        message = {
            "timestamp": timestamp,
            "user_input": user_input,
            "assistant_response": response,
            "function_used": function_used,
        }

        conversation["messages"].append(message)
        conversation["last_updated"] = timestamp

        # Save conversation
        with open(filename, "w") as f:
            json.dump(conversation, f, indent=2)

    except Exception as e:
        print(f"\033[91mError logging conversation: {str(e)}\033[0m")


def process_input(user_input):
    """Process user input and determine appropriate action."""
    try:
        # Get function and parameters from input
        function_name, param1, param2 = get_function_from_gemini(user_input)

        # Map functions to their handlers
        function_map = {
            "chat_with_gemini": chat_with_gemini,
            "open_application": open_application,
            "close_application": close_application,
            "search_and_speak": search_and_speak,
            "get_weather": get_weather,
            "set_timer": set_timer,
            "get_current_time": get_current_time,
            "remember": remember,
            "recall": recall,
            "generate_file": generate_file,
            "list_conversations": list_conversations,
            "load_conversation": load_conversation,
        }

        # Execute the appropriate function
        if function_name in function_map:
            if param2 is not None:
                response = function_map[function_name](param1, param2)
            elif param1 is not None:
                response = function_map[function_name](param1)
            else:
                response = function_map[function_name]()
        else:
            response = chat_with_gemini(user_input)

        return response

    except Exception as e:
        print(f"\033[91mError in process_input: {str(e)}\033[0m")
        return f"Error processing input: {str(e)}"


def assistant(user_input):
    """Main assistant function that processes input and manages responses."""
    try:
        # Get current time for logging
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Process the input and get response
        response = process_input(user_input)

        # Log the conversation
        log_conversation(timestamp, user_input, response)

        # Return the response without speaking (speech handled by server)
        return response

    except Exception as e:
        print(f"\033[91mError in assistant: {str(e)}\033[0m")
        return f"I encountered an error: {str(e)}"


def cleanup_resources():
    """Clean up resources before shutdown."""
    global engine, is_speaking
    try:
        is_speaking = False
        if engine:
            engine.stop()
            del engine
    except Exception as e:
        print(f"Error during cleanup: {e}")


# Initialize necessary files and directories
def initialize_files():
    """Initialize necessary files and directories."""
    try:
        # Create directories if they don't exist
        directories = ["conversations", "output"]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Initialize files if they don't exist
        files = {
            CONVERSATION_FILE: [],
            LEARNING_FILE: {},
            MEMORY_FILE: {},
            REMINDERS_FILE: [],
        }

        for file_path, default_content in files.items():
            if not os.path.exists(file_path):
                with open(file_path, "wb" if file_path.endswith(".pkl") else "w") as f:
                    if file_path.endswith(".pkl"):
                        pickle.dump(default_content, f)
                    else:
                        json.dump(default_content, f)

    except Exception as e:
        print(f"Error initializing files: {e}")


# Call initialization on import
initialize_files()

# Load memory data
try:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            memory_data = pickle.load(f)
except Exception as e:
    print(f"Error loading memory: {e}")
    memory_data = {}

# Load reminders
try:
    if os.path.exists(REMINDERS_FILE):
        with open(REMINDERS_FILE, "rb") as f:
            reminders = pickle.load(f)
except Exception as e:
    print(f"Error loading reminders: {e}")
    reminders = []


def main():
    """Main loop for the assistant."""
    global engine, is_speaking, is_listening, last_activity_time

    # Set process priority
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows
    except:
        pass

    # Enable garbage collection
    gc.enable()

    # Schedule periodic cleanup
    def periodic_cleanup():
        while True:
            cleanup_resources()
            time.sleep(300)  # Run every 5 minutes

    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()

    # Load memory and reminders
    load_memory()
    load_reminders()

    # Initialize conversation storage
    initialize_conversation_storage()

    # Load learned patterns at startup
    load_learned_patterns()

    # Register the force exit function
    atexit.register(force_exit)
    signal.signal(signal.SIGINT, lambda x, y: force_exit())
    signal.signal(signal.SIGTERM, lambda x, y: force_exit())

    print("\033[95m=== Max is ready! ===\033[0m")
    greet_user()

    # Set initial state
    is_listening = True
    last_activity_time = time.time()

    try:
        while True:
            try:
                # Check for timeout before listening
                check_timeout()

                user_input = listen()

                if user_input == "exit":
                    speak("Goodbye! Have a great day!")
                    time.sleep(1)
                    save_learned_patterns()
                    force_exit()
                elif user_input is not None:
                    assistant(user_input)

                # Small delay to prevent high CPU usage
                time.sleep(0.1)

            except Exception as e:
                print(f"\n\033[91mCritical error: {str(e)}\033[0m")
                save_learned_patterns()
                force_exit()

    except KeyboardInterrupt:
        print("\n\033[91mExiting assistant...\033[0m")
        save_learned_patterns()
        force_exit()
    except Exception as e:
        print(f"\n\033[91mUnexpected error: {str(e)}\033[0m")
        save_learned_patterns()
        force_exit()
    finally:
        save_learned_patterns()
        force_exit()


# Main Loop
if __name__ == "__main__":
    main()


def stop_speaking():
    """Stop the current speech output."""
    global is_speaking, should_pause, current_speech_thread, engine

    try:
        with engine_lock:
            should_pause = True
            is_speaking = False

            if engine:
                try:
                    engine.endLoop()
                    engine.stop()
                except:
                    pass

            if current_speech_thread and current_speech_thread.is_alive():
                current_speech_thread.join(timeout=0.5)

        return {"status": "success", "message": "Speech stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@contextmanager
def engine_lifecycle():
    """Context manager for speech engine lifecycle."""
    global engine
    try:
        if engine:
            engine.endLoop()  # End any existing loop
            engine.stop()
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        yield engine
    finally:
        if engine:
            try:
                engine.endLoop()
                engine.stop()
            except:
                pass
