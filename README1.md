*Temporal Data Parsing and Extraction: Understanding Timestamp Components*

Understanding the sample code
When you run:

python3 nn.py train model train
the sample code will call the method:

def train(model_dir, data_dir, epochs, batch_size, learning_rate):
This method iterates over the training data, uses spacy to split the text into sentences and tokens, and converts the XML annotations into token-level labels. The code will print out the first few labeled tokens so you can get an idea of what the data looks like:

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
For example, the second row of this output shows that the characters from offsets 26 to 30 were '1998', and that the XML annotations assigned this span the label 'Year'. The row also shows what the neural network will see: a single input feature, 1551, which is the index that spacy assigns the token '1998', and a single output value, 27, which is the index that the sample code assigns the label 'Year'.
