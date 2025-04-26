# cs224n-assignment-2-word2vec-solved
**TO GET THIS SOLUTION VISIT:** [CS224n Assignment #2: word2vec Solved](https://www.ankitcodinghub.com/product/cs224n-assignment-2-word2vec-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;49397&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS224n Assignment #2: word2vec Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
<h1>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Written: Understanding word2vec (23 points)</h1>
Let’s have a quick refresher on the word2vec algorithm. The key insight behind word2vec is that <em>‘a word is known by the company it keeps’</em>. Concretely, suppose we have a ‘center’ word <em>c </em>and a contextual window surrounding <em>c</em>. We shall refer to words that lie in this contextual window as ‘outside words’. For example, in Figure 1 we see that the center word <em>c </em>is ‘banking’. Since the context window size is 2, the outside words are ‘turning’, ‘into’, ‘crises’, and ‘as’.

The goal of the skip-gram word2vec algorithm is to accurately learn the probability distribution <em>P</em>(<em>O</em>|<em>C</em>). Given a specific word <em>o </em>and a specific word <em>c</em>, we want to calculate <em>P</em>(<em>O </em>= <em>o</em>|<em>C </em>= <em>c</em>), which is the probability that word <em>o </em>is an ‘outside’ word for <em>c</em>, i.e., the probability that <em>o </em>falls within the contextual window of <em>c</em>.

Figure 1: The word2vec skip-gram prediction model with window size 2

In word2vec, the conditional probability distribution is given by taking vector dot-products and applying the softmax function:

exp(<em>u</em><sup>&gt;</sup><em><sub>o </sub></em><em>v<sub>c</sub></em>)

<em>P</em>(<em>O </em>= <em>o </em>| <em>C </em>= <em>c</em>) =&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1)

P<em>w</em>∈Vocab exp(<em>u</em>&gt;<em>w</em><em>v</em><em>c</em>)

Here, <em>u<sub>o </sub></em>is the ‘outside’ vector representing outside word <em>o</em>, and <em>v<sub>c </sub></em>is the ‘center’ vector representing center word <em>c</em>. To contain these parameters, we have two matrices, <em>U </em>and <em>V </em>. The columns of <em>U </em>are all the ‘outside’ vectors <em>u<sub>w</sub></em>. The columns of <em>V </em>are all of the ‘center’ vectors <em>v<sub>w</sub></em>. Both <em>U </em>and <em>V </em>contain a vector for every <em>w </em>∈ Vocabulary.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>

Recall from lectures that, for a single pair of words <em>c </em>and <em>o</em>, the loss is given by:

<em>J</em>naive-softmax(<em>v</em><em>c,o,</em><em>U</em>) = −log<em>P</em>(<em>O </em>= <em>o</em>|<em>C </em>= <em>c</em>)<em>.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </em>(2)

Another way to view this loss is as the cross-entropy<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> between the true distribution <em>y </em>and the predicted distribution <em>y</em>ˆ. Here, both <em>y </em>and <em>y</em>ˆ are vectors with length equal to the number of words in the vocabulary. Furthermore, the <em>k<sup>th </sup></em>entry in these vectors indicates the conditional probability of the <em>k<sup>th </sup></em>word being an ‘outside word’ for the given <em>c</em>. The true empirical distribution <em>y </em>is a one-hot vector with a 1 for the true outside word <em>o</em>, and 0 everywhere else. The predicted distribution <em>y</em>ˆ is the probability distribution <em>P</em>(<em>O</em>|<em>C </em>= <em>c</em>) given by our model in equation (1).

<ul>
<li>(3 points) Show that the naive-softmax loss given in Equation (2) is the same as the cross-entropy loss between <em>y </em>and <em>y</em>ˆ; i.e., show that</li>
</ul>
1

CS 224n Assignment #2: word2vec (43 Points)

− <sup>X </sup><em>y<sub>w </sub></em>log(ˆ<em>y<sub>w</sub></em>) = −log(ˆ<em>y<sub>o</sub></em>)<em>.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </em>(3)

<em>w</em>∈<em>Vocab</em>

Your answer should be one line.

<ul>
<li>(5 points) Compute the partial derivative of <em>J</em><sub>naive-softmax</sub>(<em>v<sub>c</sub>,o,</em><em>U</em>) with respect to <em>v<sub>c</sub></em>. Please write your answer in terms of <em>y</em>, <em>y</em>ˆ, and <em>U</em>.</li>
<li>(5 points) Compute the partial derivatives of <em>J</em><sub>naive-softmax</sub>(<em>v<sub>c</sub>,o,</em><em>U</em>) with respect to each of the ‘outside’ word vectors, <em>u<sub>w</sub></em>’s. There will be two cases: when <em>w </em>= <em>o</em>, the true ‘outside’ word vector, and <em>w </em>6= <em>o</em>, for all other words. Please write you answer in terms of <em>y</em>, <em>y</em>ˆ, and <em>v<sub>c</sub></em>.</li>
<li>(3 Points) The sigmoid function is given by Equation 4:</li>
</ul>
1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>e</em><em><sup>x</sup></em>

<em>σ</em>(<em>x</em>) =&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; =&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (4)

1 + <em>e</em><sup>−</sup><em><sup>x&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </sup>e</em><em><sup>x </sup></em>+ 1

Please compute the derivative of <em>σ</em>(<em>x</em>) with respect to <em>x</em>, where <em>x </em>is a vector.

<ul>
<li>(4 points) Now we shall consider the Negative Sampling loss, which is an alternative to the Naive Softmax loss. Assume that <em>K </em>negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as <em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w<sub>K </sub></em>and their outside vectors as <em>u</em><sub>1</sub><em>,…,</em><em>u<sub>K</sub></em>. Note that <em>o /</em>∈ {<em>w</em><sub>1</sub><em>,…,w<sub>K</sub></em>}. For a center word <em>c </em>and an outside word <em>o</em>, the negative sampling loss function is given by:</li>
</ul>
<em>J</em><sub>neg-sample&nbsp;&nbsp;&nbsp; </sub>))&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (5)

for a sample <em>w</em><sub>1</sub><em>,…w<sub>K</sub></em>, where <em>σ</em>(·) is the sigmoid function.<sup>3</sup>

Please repeat parts (b) and (c), computing the partial derivatives of <em>J</em><sub>neg-sample </sub>with respect to <em>v<sub>c</sub></em>, with respect to <em>u<sub>o</sub></em>, and with respect to a negative sample <em>u<sub>k</sub></em>. Please write your answers in terms of the vectors <em>u<sub>o</sub></em>, <em>v<sub>c</sub></em>, and <em>u<sub>k</sub></em>, where <em>k </em>∈ [1<em>,K</em>]. After you’ve done this, describe with one sentence why this loss function is much more efficient to compute than the naive-softmax loss. Note, you should be able to use your solution to part (d) to help compute the necessary gradients here.

<ul>
<li>(3 points) Suppose the center word is <em>c </em>= <em>w<sub>t </sub></em>and the context window is [<em>w<sub>t</sub></em><sub>−<em>m</em></sub>, <em>…</em>, <em>w<sub>t</sub></em><sub>−1</sub>, <em>w<sub>t</sub></em>, <em>w<sub>t</sub></em><sub>+1</sub>, <em>…</em>, <em>w<sub>t</sub></em><sub>+<em>m</em></sub>], where <em>m </em>is the context window size. Recall that for the skip-gram version of word2vec, the total loss for the context window is:</li>
</ul>
<em>J</em>skip-gram(<em>v</em><em>c,w</em><em>t</em>−<em>m</em><em>,…w</em><em>t</em>+<em>m</em><em>,</em><em>U</em>) = X <em>J</em>(<em>v</em><em>c,w</em><em>t</em>+<em>j</em><em>,</em><em>U</em>)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (6)

−<em>m</em>≤<em>j</em>≤<em>m j</em>6=0

Here, <em>J</em>(<em>v<sub>c</sub>,w<sub>t</sub></em><sub>+<em>j</em></sub><em>,</em><em>U</em>) represents an arbitrary loss term for the center word <em>c </em>= <em>w<sub>t </sub></em>and outside word <em>w</em><em>t</em>+<em>j</em>. <em>J</em>(<em>v</em><em>c,w</em><em>t</em>+<em>j</em><em>,</em><em>U</em>) could be <em>J</em>naive-softmax(<em>v</em><em>c,w</em><em>t</em>+<em>j</em><em>,</em><em>U</em>) or <em>J</em>neg-sample(<em>v</em><em>c,w</em><em>t</em>+<em>j</em><em>,</em><em>U</em>), depending on your implementation.

Write down three partial derivatives:

(i) <em>∂</em><em>J</em>skip-gram(<em>v</em><em>c,w</em><em>t</em>−<em>m</em><em>,…w</em><em>t</em>+<em>m</em><em>,</em><em>U</em>)<em>/∂</em><em>U</em>

<h2>(ii) ∂Jskip-gram(vc,wt−m,…wt+m,U)/∂vc</h2>
<sup>3</sup>Note: the loss function here is the negative of what Mikolov et al. had in their original paper, because we are doing a minimization instead of maximization in our assignment code. Ultimately, this is the same objective function.

Page 2 of 3

CS 224n Assignment #2: word2vec (43 Points)

<h2>(iii) ∂Jskip-gram(vc,wt−m,…wt+m,U)/∂vw when w 6= c</h2>
Write your answers in terms of <em>∂</em><em>J</em>(<em>v<sub>c</sub>,w<sub>t</sub></em><sub>+<em>j</em></sub><em>,</em><em>U</em>)<em>/∂</em><em>U </em>and <em>∂</em><em>J</em>(<em>v<sub>c</sub>,w<sub>t</sub></em><sub>+<em>j</em></sub><em>,</em><em>U</em>)<em>/∂</em><em>v<sub>c</sub></em>. This is very simple – each solution should be one line.

<em>Once you’re done: Given that you computed the derivatives of </em><em>J</em>(<em>v<sub>c</sub>,w<sub>t</sub></em><sub>+<em>j</em></sub><em>,</em><em>U</em>) <em>with respect to all the model parameters </em><em>U and </em><em>V in parts (a) to (c), you have now computed the derivatives of the full loss function </em><em>J<sub>skip-gram </sub>with respect to all parameters. You’re ready to implement </em><em>word2vec!</em>

<h1>2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Coding: Implementing word2vec (20 points)</h1>
In this part you will implement the word2vec model and train your own word vectors with stochastic gradient descent (SGD). Before you begin, first run the following commands within the assignment directory in order to create the appropriate conda virtual environment. This guarantees that you have all the necessary packages to complete the assignment.

conda env create -f env.yml conda activate a2

Once you are done with the assignment you can deactivate this environment by running:

conda deactivate

<ul>
<li>(12 points) First, implement the sigmoid function in py to apply the sigmoid function to an input vector. In the same file, fill in the implementation for the softmax and negative sampling loss and gradient functions. Then, fill in the implementation of the loss and gradient functions for the skip-gram model. When you are done, test your implementation by running python word2vec.py.</li>
<li>(4 points) Complete the implementation for your SGD optimizer in py. Test your implementation by running python sgd.py.</li>
<li>(4 points) Show time! Now we are going to load some real data and train word vectors with everything you just implemented! We are going to use the Stanford Sentiment Treebank (SST) dataset to train word vectors, and later apply them to a simple sentiment analysis task. You will need to fetch the datasets first. To do this, run sh getdatasets.sh. There is no additional code to write for this part; just run python run.py.</li>
</ul>
<em>Note: The training process may take a long time depending on the efficiency of your implementation </em><em>(an efficient implementation takes approximately an hour). Plan accordingly!</em>

After 40,000 iterations, the script will finish and a visualization for your word vectors will appear. It will also be saved as wordvectors.png in your project directory. <strong>Include the plot in your homework write up. </strong>Briefly explain in at most three sentences what you see in the plot.

<h1>3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Submission Instructions</h1>
You shall submit this assignment on GradeScope as two submissions – one for “Assignment 2 [coding]” and another for ‘Assignment 2 [written]”:

<ul>
<li>Run the sh script to produce your assignment2.zip file.</li>
<li>Upload your zip file to GradeScope to “Assignment 2 [coding]”.</li>
<li>Upload your written solutions to GradeScope to “Assignment 2 [written]”.</li>
</ul>
Page 3 of 3

<a href="#_ftnref1" name="_ftn1">[1]</a> Assume that every word in our vocabulary is matched to an integer number <em>k</em>. <em>u<sub>k </sub></em>is both the <em>k<sup>th </sup></em>column of <em>U </em>and the ‘outside’ word vector for the word indexed by <em>k</em>. <em>v<sub>k </sub></em>is both the <em>k<sup>th </sup></em>column of <em>V </em>and the ‘center’ word vector for the word indexed by <em>k</em>. <strong>In order to simplify notation we shall interchangeably use </strong><em>k </em><strong>to refer to the word and the index-of-the-word.</strong>

<a href="#_ftnref2" name="_ftn2">[2]</a> The Cross Entropy Loss between the true (discrete) probability distribution <em>p </em>and another distribution <em>q </em>is −<sup>P</sup><em><sub>i </sub>p<sub>i </sub></em>log(<em>q<sub>i</sub></em>).
