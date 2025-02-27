# Reed-Solomon Encoding

**Base idea:** by structuring the data in a way that we can "guess" what the data was, in case it gets corrupted, just by "fixing" the structure

**Base idea expanded:** you want to communicate a set of words, a dictionary, but instead of using a full dictionary, we can use a "reduced dictionary" such that each word is as different as possible from the others.

**Example**

Dictionary; this, that, corn.

If we recieve co** (last two letters are corrupted) we know that it is corn, since it doesnt match any of the others, however if we receive th**, we dont know which word it is. To make our dictionary better, we could replace "that" with a word like "dash" to maximize the difference between each word.

The minimum number of different letters between any two words in our dictionary are called **The Minimum Hamming Distance**

Ensuring that any of the two words of the dictionary share a minimum number of letters at the same position is called **maximum** **separability**

**Example:**

If we have a 5-letter dictionary, any word must differ in 3 different positions (why this is, is shown later)

APPLE - GRAPE 	-> differ in 4 positions (valid)

TABLE - CABLE	-> differ in 3 positions (valid)

PLATE - PLANE 	-> differ in 2 positions (invalid)
