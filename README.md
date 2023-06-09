# Objectives

The learning objectives of this assignment are to:
1. build a neural network for a community task 
2. practice tuning model hyper-parameters

# Read about the CodaLab Competition

You will be participating in a class-wide competition.
The competition website is:

https://codalab.lisn.upsaclay.fr/competitions/8083?secret_key=dcf746c2-f094-446b-9c6b-00c4f0e43c67

You should visit that site and read all the details of the competition, which
include the task definition, how your model will be evaluated, the format of
submissions to the competition, etc.

# Create a CodaLab account

You must create a CodaLab account and join the competition:
1. Visit the competition website.

2. In the upper right corner of the page, you should see a "Sign Up" button.
Click that and go through the process to create an account.
**Please use your @arizona.edu account when signing up.**
Your username will be displayed publicly on a leaderboard showing everyone's
scores.
**If you wish to remain anonymous, please select a username that does not reveal
your identity.**
Your instructor will still be able to match your score with your name via your
email address, but your email address will not be visible to other students. 

3. Return to the competition website and click the "Participate" tab, where you
should be able to request to be added to the competition.

4. Wait for your instructor to manually approve your request.
This may take a day or two. 

5. You should then be able to return to the "Participate" tab and see a
"Submit / View Results" option.
That means you are fully registered for the task.

# Clone the repository

Clone the repository created by GitHub Classroom to your local machine:
```
git clone https://github.com/ua-ista-457/graduate-project-<your-username>.git
```
Note that you do not need to create a separate branch
(though you're welcome to if you so choose).
You are now ready to begin working on the assignment.

# Write your code

You should design a neural network model to perform the task described on the
CodaLab site.
You must create and train your neural network in the Keras framework that we
have been using in the class.
You should train and tune your model using the training and development data
that is already included in your GitHub Classroom repository.

**You may incorporate extra resources beyond this training data, but only if
you provide those same resources to all other students in the class by posting
the resource in the `#graduate-project` channel on the class's Slack workspace:
http://ua-ista457-fa22.slack.com**

# Using the sample code

There is some sample code in your repository from which you could start, but
you should feel free to delete that code entirely and start from scratch if
you prefer.

If you choose to use the sample code, you will need to install the anaforatools
library for parsing the XML:

    pip3 install anaforatools

You will also need to install [spacy](https://spacy.io/) and its English model:

    pip3 install spacy
    python3 -m spacy download en_core_web_md

Then you can train a very simple feed-forward neural network on the data in the
`train` folder, and use the model to make predictions on the data in the `dev`
folder and write the predictions to a `submission` folder:

     python3 nn.py train model train
     python3 nn.py predict model submission dev

This will also produce a `submission.zip` that you can upload to CodaLab.

# Understanding the sample code

When you run:
```
python3 nn.py train model train
```
the sample code will call the method:
```python
def train(model_dir, data_dir, epochs, batch_size, learning_rate):
```
This method iterates over the training data, uses spacy to split the text into
sentences and tokens, and converts the XML annotations into token-level
labels.
The code will print out the first few labeled tokens so you can get an idea of
what the data looks like:
```
  20:22    [11268]->13  '05'->Month-Of-Year'
  23:25    [ 8421]-> 6  '01'->Day-Of-Month'
  26:30    [ 1551]->27  '1998'->Year'
  31:33    [11319]-> 9  '09'->Hour-Of-Day'
  34:36    [  807]->11  '13'->Minute-Of-Hour'
  37:39    [12536]->21  '00'->Second-Of-Minute'
  70:72    [11268]->13  '05'->Month-Of-Year'
  73:75    [ 8421]-> 6  '01'->Day-Of-Month'
  76:80    [ 1551]->27  '1998'->Year'
  81:83    [11319]-> 9  '09'->Hour-Of-Day'
  84:86    [  807]->11  '13'->Minute-Of-Hour'
  87:89    [12536]->21  '00'->Second-Of-Minute'
```
For example, the second row of this output shows that the characters from
offsets 26 to 30 were `'1998'`, and that the XML annotations assigned this span
the label `'Year'`.
The row also shows what the neural network will see: a single input feature,
`1551`, which is the index that spacy assigns the token `'1998'`, and a single
output value, `27`, which is the index that the sample code assigns the label
`'Year'`.

# Extending the sample code

Building an accurate system for the task will require significant modifications
to the sample code.
Some things that may be worth trying:

1. Classifying not just one token at a time, but a series of tokens, using a
   convolutional, recurrent, or transformer network.
   The current code already groups tokens by sentences, but then flattens the
   array and works with one token at a time via `.reshape(-1, 1)`.
   For convolutional, recurrent, or transformer networks, you will likely want
   to retain the original shape and modify the code:
   ```python
   model = tf.keras.models.Sequential([...]) 
   ```
   to use your chosen architecture instead of the feedforward architecture.

2. One of the many pre-trained transformer networks provided by the
   [Huggingface transformers library](https://huggingface.co/docs/transformers/).
   If you pursue this direction, a good starting point is the
   "training in TensorFlow with the Keras API" section of
   [Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training).
   As in the item above, you will likely want spacy to break your text into
   sentences.
   But you will also need to study how tokenization, feature, and label
   formats change when using the huggingface transformers library.
   If you use this library, make sure you use the Keras/Tensorflow APIs, **not
   the PyTorch APIs**!
   If you don't have a `model.compile(...)` followed by a `model.fit(...)`, you
   are probably using the wrong APIs and will lose points as described below
   under Grading.

# Test your model predictions on the development set

To test the performance of your model, the only officially supported way is to
run your model on the development set (included in your GitHub Classroom
checkout), format your model predictions as instructed on the CodaLab site,
and upload your model's predictions on the "Participate" tab of the CodaLab
site.

Unofficially, you may use the `anafora.evaluate` package as the example code
does.
But you are **strongly** encouraged to upload your model's development set
predictions to the CodaLab site many times to make sure you have all the
formatting correct.
Otherwise, you risk trying to debug formatting issues on the test set, when
the time to submit your model predictions is much more limited.

# Submit your model predictions on the test set

When the test phase of the competition begins (consult the CodaLab site for the
exact timing, and be aware that CodaLab times are in UTC timezone), the
instructor will release the test data and update the CodaLab site to expect
predictions on the test set, rather than predictions on the development set.
You should run your model on the test data, and upload your model predictions to
the CodaLab site.
 
# Grading

You will be graded first by your model's accuracy, and second on how well your
model ranks in the competition.
If your model achieves better accuracy on the test set than the baseline model
included in this repository, you will get at least a B.
If your model achieves better accuracy on the test set than another baseline
that the instructor will reveal after the competition, you will get an A.
All models within the same letter grade will be distributed evenly across the
range, based on their rank.
So for example, the highest ranked model in the A range will get 100%, and the
lowest ranked model in the B range will get 80%.

If you use a neural network library other than Tensorflow/Keras, or an external
resource that you do not share in the `#graduate-project` channel of
http://ua-ista457-fa21.slack.com, you will lose 10% (a letter grade) from your
score.
