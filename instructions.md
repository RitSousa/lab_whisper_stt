# Compare Whisper vs Google STT

> **How you'll submit this lab**
>
> This repo is your lab. Fork it, do the work described below in your fork, then open a pull
> request back into this repository. An AI reviewer will check your PR against `rubric.md` and
> leave feedback directly on the PR. See `README.md` for the full workflow.

**Scenario**  
Your client needs to evaluate Whisper's transcription accuracy before deploying it at scale for their meeting transcription needs. They want you to assess how well Whisper performs across different audio samples, measure accuracy using Word Error Rate (WER), analyze costs, and provide recommendations based on performance and budget. You'll transcribe audio with Whisper, create human-verified ground truth transcriptions, calculate WER to measure accuracy, and work with a partner to compare results across different voices and audio samples.

**Learning Objectives**
- [ ] Record audio for transcription testing
- [ ] Transcribe audio using Whisper API
- [ ] Create human-verified ground truth transcriptions
- [ ] Calculate Word Error Rate (WER) using jiwer library
- [ ] Analyze transcription accuracy differences
- [ ] Analyze Whisper API costs
- [ ] Generate a comprehensive evaluation report
- [ ] Make recommendations based on performance and cost
- [ ] Collaborate with a partner to compare results across different audio samples
- [ ] Analyze patterns and differences in transcription performance

**Estimated Time:** 200-240 minutes (includes pair work and comparison)

**Prerequisites:**
- [ ] Completion of Lab M1.07 (Whisper STT Implementation)
- [ ] Basic understanding of API usage
- [ ] Familiarity with evaluation metrics
- [ ] Understanding of cost analysis

---

## Introduction

Welcome to the Whisper STT Evaluation lab! In this exercise, you'll conduct a comprehensive evaluation of OpenAI's Whisper API to assess its transcription accuracy and performance. This is a real-world scenario where you need to evaluate an AI service before deploying it at scale.

**What you'll build:**
- Whisper API transcription system
- WER calculation system using ground truth
- Cost analysis tool
- Comprehensive evaluation report

**Why this matters:**
Evaluating AI service performance is crucial for:
- Accuracy assessment (measuring real-world performance)
- Cost optimization (understanding pricing for your use case)
- Quality assurance (ensuring the service meets requirements)
- Decision making (data-driven deployment decisions)

**Success criteria:**
- [ ] Successfully transcribe audio with Whisper
- [ ] Create accurate ground truth transcriptions
- [ ] Calculate WER accurately
- [ ] Analyze Whisper API costs
- [ ] Generate a comprehensive evaluation report
- [ ] Provide clear recommendations

---

## Background Story

Your client processes hundreds of hours of meeting recordings monthly and is considering using Whisper for transcription. Before deploying at scale, they need to:

1. **Assess accuracy** - Understand how well Whisper performs on their audio types
2. **Evaluate costs** - Determine if Whisper pricing fits their budget
3. **Identify patterns** - Understand how different voices, accents, and audio qualities affect performance
4. **Make informed decisions** - Use data to decide if Whisper meets their needs

You've been tasked with:
- Recording your own audio sample for testing
- Transcribing your audio with Whisper
- Creating a human-verified ground truth transcription
- Measuring accuracy using Word Error Rate (WER)
- Analyzing costs
- Comparing results with a partner who has tested with their own audio
- Providing recommendations based on comprehensive analysis

This will help them make an informed decision about deploying Whisper. By working in pairs and comparing results across different voices and audio samples, you'll gain deeper insights into how Whisper performs in real-world scenarios.

---

## Pair Work Structure

**Important:** This lab is designed for pair work. You'll work with a partner to compare results and gain deeper insights.

**How it works:**
1. **Independent Work:** Both you and your partner will:
   - Record your own audio file (each person records separately)
   - Transcribe your audio with Whisper
   - Create a human-verified ground truth transcription
   - Calculate WER and perform cost analysis
   - Generate your own evaluation report

2. **Pair Comparison:** After completing your individual analysis:
   - Share your audio files and results with your partner
   - Compare transcriptions from both students' audio
   - Compare WER results and identify patterns
   - Discuss how Whisper performed for each person
   - Document differences and insights in a pair comparison summary

**Why pair work?**
- Different voices, accents, and speaking styles test Whisper more thoroughly
- Comparing results helps identify patterns and edge cases
- Collaborative analysis leads to better understanding of Whisper's strengths and weaknesses
- Real-world projects often involve team collaboration

**Sharing files:**
- Share your audio file with your partner (via email, cloud storage, or messaging)
- Share your transcription results and WER calculations
- Compare notes on your findings

---

## Step-by-Step Instructions

### Step 1: Recording Your Audio

**Objective:** Record an audio file that you'll use for transcription evaluation.

**Audio Requirements:**
- **Duration:** 30-60 seconds of speech
- **Content:** Speak clearly about a topic of your choice (e.g., describe your day, explain a concept, tell a story)
- **Format:** WAV, MP3, or M4A (will be converted if needed)
- **Quality:** Clear audio with minimal background noise (for baseline testing)

**Option 1: Recording with Python (Recommended)**

You can use the recording functions from the `speech_recognition_breaking_points.ipynb` notebook as a reference. Here's how:

1. **Open the notebook:** Navigate to `course/module-1/submodule-4-speech-recognition/content/speech_recognition_breaking_points.ipynb`

2. **Use the recording functions:** The notebook contains helpful functions:
   - `record_audio()` - Records audio with a countdown
   - `save_recording()` - Saves audio to WAV file
   - `visualize_waveform()` - Visualizes the recorded audio

3. **Quick recording script:**
```python
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time

def record_audio(duration=30, sample_rate=16000):
    """Record audio for specified duration"""
    print(f"🎤 Recording for {duration} seconds...")
    print("Get ready...")
    time.sleep(2)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("🔴 RECORDING! Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    
    print("✅ Recording complete!")
    return audio.flatten(), sample_rate

# Record your audio
audio, sr = record_audio(duration=45)  # 45 seconds

# Save to file
audio = np.clip(audio, -1, 1)
audio_int16 = (audio * 32767).astype(np.int16)
wavfile.write('my_audio_recording.wav', sr, audio_int16)
print("✅ Saved to 'my_audio_recording.wav'")
```

**Required packages for Python recording:**
```bash
pip install sounddevice numpy scipy
```

**Option 2: Recording on Your Phone**

If you prefer to record on your phone:

1. **Use your phone's voice recorder app:**
   - iPhone: Use the built-in Voice Memos app
   - Android: Use the built-in Recorder app or download a free recording app

2. **Recording tips:**
   - Speak clearly and at a normal pace
   - Record in a quiet environment
   - Aim for 30-60 seconds of speech
   - Hold the phone at a comfortable distance (not too close or far)

3. **Transfer to your computer:**
   - **Email:** Send the audio file to yourself
   - **Cloud storage:** Upload to Google Drive, Dropbox, or iCloud
   - **USB cable:** Connect phone to computer and transfer file
   - **AirDrop (Mac/iOS):** Use AirDrop to transfer wirelessly

4. **File format:**
   - Most phones record in M4A or MP3 format
   - Both formats work with Whisper
   - If needed, convert to WAV using online converters or `pydub` library

**After recording:**
- [ ] Save your audio file with a clear name (e.g., `student1_audio.wav`)
- [ ] Verify the audio plays correctly
- [ ] Share the file with your partner
- [ ] Keep a copy for your own analysis

**Checkpoint:** You should have an audio file ready for transcription.

---

### Step 2: Transcribing with Whisper

**Objective:** Transcribe your audio file using Whisper API.

This step uses the same code as your previous Whisper transcription code from Lab M1.07.

**What to do:**
1. Use your previous Whisper transcription code from Lab M1.07
2. Transcribe your audio file
3. Save the transcription result for WER calculation

**Reference:** Review Lab M1.07 for Whisper API setup and transcription code.

**Expected outcome:** You should have a Whisper transcription of your audio.

**Checkpoint:** Verify that Whisper transcription is complete and saved.

---

### Step 3: Creating Ground Truth Transcription

**Objective:** Create a human-verified ground truth transcription for WER calculation.

**What to do:**
1. Use the Whisper transcription as a starting point
2. Listen to your audio recording carefully
3. Manually correct any errors in the transcription
4. Verify that the transcription accurately reflects what was said
5. Save the corrected transcription as your ground truth

**Note:** The ground truth transcription should be as accurate as possible. Listen to the audio multiple times if needed to catch all words, including filler words, hesitations, and corrections.

**Expected outcome:** You should have a human-verified ground truth transcription saved.

**Checkpoint:** Verify that your ground truth transcription accurately reflects the audio content.

---

### Step 4: Calculating Word Error Rate (WER)

**Objective:** Calculate WER to measure Whisper transcription accuracy.

**What to do:**
1. Use your human-verified ground truth transcription as the reference
2. Calculate WER using jiwer library
3. Analyze the error breakdown (substitutions, insertions, deletions)
4. Interpret the results

**Code template:**
```python
import jiwer
import json

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    
    WER = (Substitutions + Insertions + Deletions) / Total Words in Reference
    
    Parameters:
    - reference: Ground truth transcription
    - hypothesis: Predicted transcription (from Whisper)
    
    Returns:
    - Dictionary with WER metrics
    """
    # Calculate WER using jiwer
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    reference_clean = transformation(reference)
    hypothesis_clean = transformation(hypothesis)
    
    # Calculate metrics
    measures = jiwer.compute_measures(
        truth=reference_clean,
        hypothesis=hypothesis_clean
    )
    
    wer = measures['wer']
    substitutions = measures['substitutions']
    insertions = measures['insertions']
    deletions = measures['deletions']
    hits = measures['hits']
    
    # Calculate word counts
    ref_words = len(reference_clean.split())
    hyp_words = len(hypothesis_clean.split())
    
    return {
        "wer": wer,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "hits": hits,
        "reference_words": ref_words,
        "hypothesis_words": hyp_words,
        "accuracy": 1 - wer  # Accuracy = 1 - WER
    }

def evaluate_whisper_accuracy(whisper_text, ground_truth_text):
    """
    Evaluate Whisper transcription accuracy against ground truth.
    
    Parameters:
    - whisper_text: Whisper transcription
    - ground_truth_text: Human-verified ground truth transcription
    
    Returns:
    - Dictionary with evaluation results
    """
    print("\n" + "="*50)
    print("EVALUATING WHISPER ACCURACY")
    print("="*50)
    
    wer_metrics = calculate_wer(ground_truth_text, whisper_text)
    
    print(f"\nWord Error Rate (WER): {wer_metrics['wer']:.4f} ({wer_metrics['wer']*100:.2f}%)")
    print(f"Accuracy: {wer_metrics['accuracy']:.4f} ({wer_metrics['accuracy']*100:.2f}%)")
    print(f"\nError Breakdown:")
    print(f"  Substitutions: {wer_metrics['substitutions']}")
    print(f"  Insertions: {wer_metrics['insertions']}")
    print(f"  Deletions: {wer_metrics['deletions']}")
    print(f"  Correct words: {wer_metrics['hits']}")
    print(f"\nWord Counts:")
    print(f"  Reference words: {wer_metrics['reference_words']}")
    print(f"  Hypothesis words: {wer_metrics['hypothesis_words']}")
    
    return wer_metrics

# Calculate WER
if whisper_result and ground_truth_text:
    wer_results = evaluate_whisper_accuracy(
        whisper_result['text'],
        ground_truth_text
    )
    
    # Save WER results
    with open("wer_results.json", 'w') as f:
        json.dump(wer_results, f, indent=2)
    
    print("\n✓ WER evaluation saved!")
else:
    print("⚠ Need both Whisper transcription and ground truth to calculate WER")
```

**Expected outcome:** You should see WER values and detailed error breakdowns.

**Checkpoint:** Verify that WER is calculated correctly.

---

### Step 5: Cost Analysis

**Objective:** Analyze Whisper API costs for different usage scenarios.

**What to do:**
1. Calculate costs based on audio duration
2. Research current Whisper API pricing
3. Estimate costs for different volumes
4. Consider cost optimization strategies

**Whisper API Pricing :**
- Check current pricing on [OpenAI Platform](https://platform.openai.com/docs/pricing)
- Pricing is typically per minute of audio processed
- Consider any free tier or volume discounts

**Expected outcome:** You should calculate cost estimates for different usage scenarios.

**Checkpoint:** Verify that cost calculations are correct and based on current pricing.

---

### Step 6: Generating Evaluation Report

**Objective:** Create a comprehensive 1-page evaluation report.

**What to do:**
1. Compile all evaluation data
2. Create a formatted report
3. Include recommendations
4. Export as PDF or markdown

**Report sections:**
- **Accuracy Analysis:** WER results and error breakdown
- **Cost Analysis:** Pricing for different usage scenarios
- **Performance Insights:** Observations about Whisper's performance
- **Recommendations:** Whether to deploy Whisper and under what conditions

**Expected outcome:** You should have a simple evaluation report.

**Checkpoint:** Verify that the report includes all necessary sections.

---

### Step 7: Comparing Results with Your Partner

**Objective:** Compare your results with your partner's results to identify patterns and gain deeper insights.

**What to do:**
1. **Share your results:**
   - Share your audio file with your partner
   - Share your Whisper transcription
   - Share your WER calculations
   - Share your cost analysis

2. **Compare transcriptions:**
   - Look at how Whisper transcribed your partner's audio
   - Compare accuracy between your audio and your partner's audio
   - Note any differences in how Whisper handled different voices

3. **Compare WER results:**
   - Which audio had better accuracy?
   - What factors might explain differences? (voice characteristics, accent, speaking pace, audio quality, etc.)
   - Are there patterns in the types of errors Whisper makes?

4. **Compare cost analyses:**
   - Are the cost calculations similar?
   - Discuss cost optimization strategies

5. **Document your findings:**
   - Create a pair comparison summary
   - Note any interesting patterns or surprises
   - Discuss Whisper's strengths and weaknesses
   - Include insights about how different voices/audio affect transcription quality
   - Provide joint recommendations based on both analyses

**Discussion prompts:**
- How did Whisper perform overall? Was it consistent across both audio samples?
- Were there any transcription errors that surprised you?
- How did voice characteristics (pitch, pace, accent) affect transcription accuracy?
- What factors beyond accuracy and cost should be considered when deploying Whisper?
- Would you recommend Whisper for production use? Why or why not?

**Expected outcome:** You should have a pair comparison summary documenting your findings and insights.

**Checkpoint:** Verify that you've compared all aspects and documented your findings.

---

## Submission Guidelines

### What to Submit

**Required deliverables:**
- [ ] Your complete `whisper_evaluation.py` file with:
  - Whisper integration
  - WER calculation
  - Cost analysis
  - Report generation
- [ ] Your audio recording file (the one you recorded)
- [ ] Evaluation report (1 page) including:
  - Accuracy analysis (WER)
  - Cost analysis
  - Performance insights
  - Recommendations
- [ ] Pair comparison summary (1 page) including:
  - Comparison of transcriptions from both students' audio
  - Comparison of WER results
  - Discussion of patterns and differences
  - Insights about how different voices/audio affect transcription
  - Joint recommendations based on both analyses
- [ ] Screenshots or output showing:
  - Whisper transcription (for your audio)
  - WER calculations
  - Cost analysis
  - Final report
- [ ] JSON files with:
  - Transcription results
  - WER metrics
  - Cost analysis data

### How to Submit

**Upload your work:**
1. **For code files:** Upload your Python script
2. **For reports:** Upload your evaluation report (PDF or text)
3. **For data files:** Upload JSON results
4. **For screenshots:** Upload PNG or JPG images

**Important notes:**
- Include your analysis and reasoning
- Show clear evaluation findings
- Document your methodology
- Explain your recommendations
- The pair comparison summary should be a collaborative document (both partners can submit the same summary or individual summaries that reference each other's work)

**Due date:** Check with your instructor for the specific due date.

---

## Troubleshooting

**Common issues and solutions:**

**Issue 1: "Audio encoding not supported"**
- **Solution:** 
  - Convert audio to supported format (WAV, FLAC, MP3, M4A, etc.)
  - Check sample rate (16kHz recommended)
  - Use pydub to convert if needed
  - If recording on phone, ensure the file format is compatible

**Issue 2: "Can't record audio with Python"**
- **Solution:**
  - Check that sounddevice is installed: `pip install sounddevice`
  - Verify microphone permissions on your system
  - On Linux, you may need: `sudo apt-get install portaudio19-dev python3-pyaudio`
  - On Mac, you may need to grant microphone permissions in System Preferences
  - On Windows, check microphone settings in Control Panel
  - Alternative: Use phone recording and transfer the file

**Issue 3: "WER calculation errors"**
- **Solution:** 
  - Ensure both transcriptions are in same language
  - Clean text before calculating WER
  - Check that jiwer is installed correctly: `pip install jiwer`
  - Verify that ground truth and Whisper transcription are both strings

**Issue 4: "Cost calculations seem wrong"**
- **Solution:** 
  - Verify current pricing on OpenAI Platform (prices may change)
  - Check for any free tier limits
  - Account for different pricing tiers if applicable
  - Double-check your duration calculations (seconds vs minutes)

**Issue 5: "Report generation fails"**
- **Solution:** 
  - Check that all required data is available
  - Handle missing data gracefully
  - Verify file paths and permissions
  - Ensure all variables are defined before generating report

---

## Additional Resources

**Audio Recording:**
- **Reference Notebook:** `course/module-1/submodule-4-speech-recognition/content/speech_recognition_breaking_points.ipynb` - Contains example recording functions using Python
- **Python Recording Libraries:**
  - [sounddevice documentation](https://python-sounddevice.readthedocs.io/)
  - [scipy.io.wavfile documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html)
- **Phone Recording Apps:**
  - iPhone: Built-in Voice Memos app
  - Android: Built-in Recorder app or [Voice Recorder](https://play.google.com/store/apps/details?id=com.coffeebeanventures.smartrecorder)
- **Audio Conversion Tools:**
  - Online: [CloudConvert](https://cloudconvert.com/), [Online Audio Converter](https://online-audio-converter.com/)
  - Python: `pydub` library for programmatic conversion

**Reference materials:**
- [OpenAI Whisper API Documentation](https://platform.openai.com/docs/guides/speech-to-text)
- [WER Calculation with jiwer](https://github.com/jitsi/jiwer)
- [Word Error Rate Explained](https://en.wikipedia.org/wiki/Word_error_rate)

**Tools and software:**
- [OpenAI Platform](https://platform.openai.com/) - API dashboard and pricing
- [OpenAI Pricing](https://openai.com/api/pricing/) - Current pricing information

**Further reading:**
- [Evaluating Speech Recognition Systems](https://www.researchgate.net/publication/221478323_Evaluating_Speech_Recognition_Systems)
- [Speech Recognition Best Practices](https://platform.openai.com/docs/guides/speech-to-text/prompting)

---

## To-Do List

- [ ] Complete Step 1: Recording Your Audio
- [ ] Share your audio file with your partner
- [ ] Complete Step 2: Transcribing with Whisper
- [ ] Complete Step 3: Creating Ground Truth Transcription
- [ ] Complete Step 4: Calculating Word Error Rate (WER)
- [ ] Complete Step 5: Cost Analysis
- [ ] Complete Step 6: Generating Evaluation Report
- [ ] Complete Step 7: Comparing Results with Your Partner
- [ ] Review and refine your report
- [ ] Create pair comparison summary
- [ ] Submit your work

---

## Learning Reflection

After completing this lab, reflect on what you've learned:

1. **Evaluation Metrics:** Why is WER useful for measuring transcription accuracy?
2. **Ground Truth:** Why is human-verified ground truth important for evaluation?
3. **Cost Analysis:** How do pricing models affect service deployment decisions?
4. **Performance Patterns:** What factors affect transcription accuracy?
5. **Decision Making:** How do you make data-driven decisions about AI service deployment?
6. **Real-world Application:** How would you apply this evaluation process to other AI services?

**Key takeaways:**
- Evaluating AI services requires accurate ground truth and proper metrics
- WER is a standard metric for evaluating transcription quality
- Cost analysis must consider usage volume and pricing tiers
- Different audio characteristics (voice, accent, quality) affect performance
- Data-driven decisions require comprehensive evaluation
- Reports should clearly communicate findings and recommendations

These skills are essential for evaluating and deploying AI services in real-world projects.

---

*Excellent work on completing the Whisper evaluation! You've learned how to evaluate AI services comprehensively and make data-driven recommendations - skills that are invaluable in AI consulting and development.*
