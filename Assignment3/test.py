import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import pretty_midi
import os
import re
import random
from tqdm import tqdm
from gensim.models import Word2Vec

# %%
# For reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")


# %% md
# Data Preparation
# %% md
## 1. Data Loading and Preprocessing
# %%
# Load lyrics data
def load_lyrics_data(file_path):
    """Load lyrics data from CSV file."""
    df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2])
    df.columns = ['artist', 'song', 'lyrics']
    return df


def is_problematic(row):
    # List of known problematic files (patterns to match in filenames)
    problematic_patterns = [
        ('beastie boys', 'girls'),
        ('billy joel', 'movin\' out'),
        ('billy joel', 'pressure'),
        ('dan fogelberg', 'leader of the band'),
        ('brian mcknight', 'on the down low'),
        ('aaron neville', 'tell it like it is'),
    ]

    return (row['artist'].lower(), row['song'].lower()) in problematic_patterns


train_lyrics = load_lyrics_data('data/lyrics_train_set.csv')
train_lyrics = train_lyrics[~train_lyrics.apply(is_problematic, axis=1)].reset_index(drop=True)

test_lyrics = load_lyrics_data('data/lyrics_test_set.csv')

# Display sample of the data
print("Training data shape:", train_lyrics.shape)
print("Test data shape:", test_lyrics.shape)


# %%
# Function to load and extract features from MIDI files
def load_simple_midi_features(midi_file_path):
    """Extract features from a MIDI file using pretty_midi."""
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Basic MIDI features
    features = {}

    # Tempo information
    tempo_changes = midi_data.get_tempo_changes()
    features['tempo'] = np.mean(tempo_changes[1])

    # Extract note features from the first instrument (assuming melody)
    notes = []

    for inst in midi_data.instruments:
        for note in inst.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'duration': note.end - note.start,
                'velocity': note.velocity
            })
        # Convert to numpy arrays for easier manipulation
        if notes:
            pitches = np.array([note['pitch'] for note in notes])
            durations = np.array([note['duration'] for note in notes])
            velocities = np.array([note['velocity'] for note in notes])
            features['avg_pitch'] = np.mean(pitches)
            features['pitch_std'] = np.std(pitches)
            features['pitch_range'] = np.max(pitches) - np.min(pitches)
            features['avg_duration'] = np.mean(durations)
            features['avg_velocity'] = np.mean(velocities)
            features['note_density'] = len(notes) / midi_data.get_end_time()
            # Store the sequence of notes (for detailed melody representation)
            features['note_sequence'] = [
                {'pitch': note.pitch,
                 'start': note.start,
                 'duration': note.end - note.start,
                 'velocity': note.velocity}
                for note in inst.notes
            ]

    return features


# %%
# Test MIDI feature extraction on a sample file
midi_files_dir = 'data/midi_files'
sample_midi_file = os.path.join(midi_files_dir, os.listdir(midi_files_dir)[0])
print(f"Testing feature extraction on: {sample_midi_file}")
sample_features = load_simple_midi_features(sample_midi_file)

for key, value in sample_features.items():
    if key != 'note_sequence':
        print(f"{key}: {value}")
print(f"Number of notes in sequence: {len(sample_features['note_sequence'])}")


# %% md
## 2. Data Preprocessing
# %%
# Process lyrics into tokenized format
def preprocess_lyrics(lyrics_series):
    """Preprocess and tokenize lyrics."""
    all_lyrics = []
    for lyric in lyrics_series:
        # Convert to lowercase and split into lines
        lyric = str(lyric).lower()
        lines = lyric.split('&')

        # Process each line
        for line in lines:
            # Clean and tokenize
            # tokens = re.findall(r'\b\w+\b', line)
            tokens = re.findall(r"\b\w+(?:'\w+)*\b", line)
            if tokens:  # Skip empty lines
                all_lyrics.append(tokens)

    return all_lyrics


# Process train and test lyrics
train_tokenized = preprocess_lyrics(train_lyrics.iloc[:, 2])
test_tokenized = preprocess_lyrics(test_lyrics.iloc[:, 2])

print(f"Number of lines in training set: {len(train_tokenized)}")
print(f"Number of lines in test set: {len(test_tokenized)}")


# %%
# Build vocabulary
def build_vocabulary(tokenized_lyrics, min_freq=2):
    """Build vocabulary from tokenized lyrics."""
    word_counts = Counter()
    for line in tokenized_lyrics:
        word_counts.update(line)

    vocab = {'<PAD>': 0, '<NEXT LINE>': 1, '<START>': 2, '<END>': 3}
    idx = 4

    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab, word_counts


vocab, word_counts = build_vocabulary(train_tokenized, min_freq=0)
print(f"Vocabulary size: {len(vocab)}")
print(f"Top 10 most common words: {word_counts.most_common(10)}")
# %%
print("Training custom Word2Vec model on lyrics dataset...")
# You can adjust these parameters based on your needs
embedding_dim = 300
window_size = 5
min_count = 1  # Can be set to match your min_freq in build_vocabulary
workers = 4  # Number of CPU cores to use

# Train the model
custom_w2v_model = Word2Vec(
    sentences=train_tokenized,
    vector_size=embedding_dim,
    window=window_size,
    min_count=min_count,
    workers=workers,
    sg=1,  # Skip-gram model (use CBOW with sg=0)
    epochs=20
)

print(f"Custom Word2Vec model trained with vector size: {custom_w2v_model.vector_size}")
print(f"Vocabulary size in Word2Vec model: {len(custom_w2v_model.wv.key_to_index)}")


# Create embedding matrix for our vocabulary
def create_embedding_matrix(vocab, embedding_model, embedding_dim=300):
    """Create embedding matrix for words in vocabulary."""
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        # Use the word vectors from your custom model
        if word in embedding_model.wv:
            embedding_matrix[idx] = embedding_model.wv[word]
        elif word not in ['<PAD>', '<NEXT LINE>', '<START>', '<END>'] and word.lower() in embedding_model.wv:
            embedding_matrix[idx] = embedding_model.wv[word.lower()]

    # Initialize special tokens with random values
    for special_token in ['<NEXT LINE>', '<START>', '<END>']:
        embedding_matrix[vocab[special_token]] = np.random.uniform(-0.25, 0.25, embedding_dim)

    return embedding_matrix


embedding_matrix = create_embedding_matrix(vocab, custom_w2v_model)
print(f"Embedding matrix shape: {embedding_matrix.shape}")


# %% md
## 3. Match Lyrics with MIDI Files
# %%
# Create a mapping between lyrics and midi files
def create_lyrics_midi_mapping(lyrics_df, midi_dir):
    """Create mapping between lyrics and MIDI files."""
    mapping = []

    # Get list of MIDI files
    midi_files = os.listdir(midi_dir)

    # For each song in the lyrics dataset, find the matching MIDI file
    for idx, row in lyrics_df.iterrows():
        artist = row['artist'].strip().lower().replace(' ', '_')
        song = row['song'].strip().lower().replace(' ', '_')
        lyrics = row['lyrics']

        # Look for matching MIDI file
        for midi_file in midi_files:
            midi_name = midi_file.lower()
            if artist in midi_name and song in midi_name:
                mapping.append({
                    'artist': artist,
                    'song': song,
                    'lyrics': lyrics,
                    'midi_file': os.path.join(midi_dir, midi_file)
                })
                break

    return mapping


# Create mapping for training and test data
train_mapping = create_lyrics_midi_mapping(train_lyrics, midi_files_dir)
test_mapping = create_lyrics_midi_mapping(test_lyrics, midi_files_dir)

print(f"Found matching MIDI files for {len(train_mapping)} out of {len(train_lyrics)} training songs")
print(f"Found matching MIDI files for {len(test_mapping)} out of {len(test_lyrics)} test songs")


class LyricsMelodySequenceDataset(Dataset):
    def __init__(self, mappings, vocab, max_sequence_length=50, context_size=3):
        self.mappings = mappings
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length
        self.context_size = context_size
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.data = []

        self.process_all_data()

    def process_all_data(self):

        for mapping in tqdm(self.mappings, desc="Processing advanced data"):

            midi = pretty_midi.PrettyMIDI(mapping["midi_file"])
            total_dur = midi.get_end_time()
            raw_lines = str(mapping["lyrics"]).lower().split("&")

            # 1) compute per‑line durations & tokenized lines
            line_durations, tokenized_lines = allocate_line_durations(raw_lines, total_dur)

            # 2) begin global containers
            full_tokens = ["<START>"] * self.context_size

            dummy_feature = {
                "pitch_mean": 0,
                "pitch_std": 0,
                "velocity_mean": 0,
                "duration_mean": 0,
                "note_density": 0,
                "local_notes": [],
            }

            all_word_features = [("<START>", dummy_feature.copy()) for _ in range(self.context_size)]

            # 3) walk through lines, keeping an absolute‑time cursor
            current_t = 0.0
            for tokens, ln_dur in zip(tokenized_lines, line_durations):
                if not tokens:  # empty line – just advance the clock
                    current_t += ln_dur
                    continue

                word_timings = estimate_word_durations(tokens, ln_dur, current_t)
                feats = extract_time_based_features(midi, word_timings)

                full_tokens.extend(tokens)
                all_word_features.extend(feats)

                # explicit line break token
                full_tokens.append("<NEXT LINE>")
                all_word_features.append(feats[-1])  # reuse last feat

                current_t += ln_dur  # advance clock

            # 4) final END token
            full_tokens.append("<END>")
            all_word_features.append(("<END>", all_word_features[-1][1]))  # same dummy

            # 5) slide a context window to build examples
            for i in range(self.context_size, len(full_tokens)):
                ctx_w = full_tokens[i - self.context_size:i]
                tgt_w = full_tokens[i]
                ctx_f = [all_word_features[j][1] for j in range(i - self.context_size, i)]
                tgt_f = all_word_features[i][1]

                self.data.append({
                    "context_words": ctx_w,
                    "context_features": ctx_f,
                    "target_word": tgt_w,
                    "target_features": tgt_f,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert words to indices
        context_indices = [self.vocab[w] for w in item['context_words']]
        target_index = self.vocab[item['target_word']]

        # Extract relevant features
        context_features = []
        for features in item['context_features']:
            # Create a fixed-size feature vector for each context word
            feature_vector = np.array([
                features['pitch_mean'] / 127.0,  # Normalize pitch
                features['pitch_std'] / 12.0,  # Normalize std
                features['velocity_mean'] / 127.0,  # Normalize velocity
                features['duration_mean'],  # Duration in seconds
                features['note_density'],  # Notes per second
                len(features['local_notes']),  # Number of notes
            ])
            context_features.append(feature_vector)

        # Target features
        target_feature_vector = np.array([[
            item['target_features']['pitch_mean'] / 127.0,
            item['target_features']['pitch_std'] / 12.0,
            item['target_features']['velocity_mean'] / 127.0,
            item['target_features']['duration_mean'],
            item['target_features']['note_density'],
            len(item['target_features']['local_notes']),
        ]])

        return {
            'context_words': torch.tensor(context_indices, dtype=torch.long),
            'context_features': torch.tensor(np.array(context_features), dtype=torch.float),
            'target_word': torch.tensor(target_index, dtype=torch.long),
            'target_features': torch.tensor(target_feature_vector, dtype=torch.float)
        }


def allocate_line_durations(lines, total_dur):
    tokenized_lines = [re.findall(r"\b\w+(?:'\w+)*\b", ln) for ln in lines]
    letter_counts = [sum(len(tok) for tok in toks) for toks in tokenized_lines]
    tot_letters = sum(letter_counts)

    durations = [(cnt / tot_letters) * total_dur for cnt in letter_counts]
    return durations, tokenized_lines


# Helper functions for the advanced dataset and model
def estimate_word_durations(
        tokens, line_duration, start_offset=0.0, word_overhead=0.1):
    total_letters = sum(len(w) for w in tokens)
    total_overhead = word_overhead * len(tokens)
    time_per_letter = (line_duration - total_overhead) / total_letters

    timings, t = [], start_offset
    for w in tokens:
        dur = len(w) * time_per_letter + word_overhead
        timings.append((w, t, t + dur))
        t += dur
    return timings


def extract_time_based_features(midi_data, word_timings):
    word_features = []

    for word, start_time, end_time in word_timings:
        features = {
            'local_notes': [],
        }

        context_window = 3  # seconds
        start_time = start_time - context_window
        end_time = end_time + context_window

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note_overlaps = note.start < end_time and note.end > start_time
                if note_overlaps:
                    features['local_notes'].append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'duration': note.end - note.start,
                        'overlap': min(note.end, end_time) - max(note.start, start_time)
                    })

        if features['local_notes']:
            features['pitch_mean'] = np.mean([n['pitch'] for n in features['local_notes']])
            features['pitch_std'] = np.std([n['pitch'] for n in features['local_notes']])
            features['velocity_mean'] = np.mean([n['velocity'] for n in features['local_notes']])
            features['duration_mean'] = np.mean([n['duration'] for n in features['local_notes']])
            features['note_density'] = len(features['local_notes'])
        else:
            print(f'WARNING empty local notes for {word}')
            # fallback defaults for empty notes
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['velocity_mean'] = 0
            features['duration_mean'] = 0
            features['note_density'] = 0

        word_features.append((word, features))

    return word_features


# Create the advanced dataset
print("Creating advanced dataset...")
context_size = 3  # Number of context words
advanced_dataset = LyricsMelodySequenceDataset(train_mapping, vocab, context_size=context_size)

# Split into train and validation
adv_train_size = int(0.8 * len(advanced_dataset))
adv_val_size = len(advanced_dataset) - adv_train_size
adv_train_dataset, adv_val_dataset = torch.utils.data.random_split(
    advanced_dataset, [adv_train_size, adv_val_size])

# Create data loaders
adv_train_loader = DataLoader(adv_train_dataset, batch_size=64, shuffle=True)
adv_val_loader = DataLoader(adv_val_dataset, batch_size=64)

print(f"Advanced train dataset size: {len(adv_train_dataset)}")
print(f"Advanced validation dataset size: {len(adv_val_dataset)}")
