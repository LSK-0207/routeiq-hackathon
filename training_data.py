# training_data.py
"""
Labeled training data for the RouteIQ category classifier.

Labels are the 8 real Track 1 capability categories, confirmed from the
hackathon reveal document:

  factual_knowledge          - explaining concepts, definitions, how things work
  mathematical_reasoning     - multi-step arithmetic, percentages, word problems, projections
  sentiment_classification   - labelling sentiment and justifying the classification
  text_summarization         - condensing passages to a specific format or length constraint
  named_entity_recognition   - extracting and labelling entities (person, org, location, date)
  code_debugging              - identifying bugs in code snippets, providing corrected implementations
  logical_reasoning           - constraint-based puzzles where all conditions must be satisfied
  code_generation              - writing correct, well-structured functions from a spec

This replaces the old local/remote-medium/remote-heavy complexity-tier
labels used in the earlier hybrid-routing architecture -- that taxonomy no
longer applies since Track 1 requires all inference to go through Fireworks.
"""

TRAINING_DATA = [

    # =========================================================================
    # factual_knowledge
    # =========================================================================
    {"prompt": "What is photosynthesis?", "label": "factual_knowledge"},
    {"prompt": "Explain how vaccines work.", "label": "factual_knowledge"},
    {"prompt": "What causes the seasons to change?", "label": "factual_knowledge"},
    {"prompt": "Define inflation in economics.", "label": "factual_knowledge"},
    {"prompt": "What is the difference between weather and climate?", "label": "factual_knowledge"},
    {"prompt": "Explain how a car engine works.", "label": "factual_knowledge"},
    {"prompt": "What is DNA and what does it do?", "label": "factual_knowledge"},
    {"prompt": "What is the capital of Japan?", "label": "factual_knowledge"},
    {"prompt": "Explain what a black hole is.", "label": "factual_knowledge"},
    {"prompt": "What is the difference between a virus and bacteria?", "label": "factual_knowledge"},
    {"prompt": "How does the human immune system work?", "label": "factual_knowledge"},
    {"prompt": "What is supply and demand?", "label": "factual_knowledge"},
    {"prompt": "Explain how the internet works at a basic level.", "label": "factual_knowledge"},
    {"prompt": "What is the greenhouse effect?", "label": "factual_knowledge"},
    {"prompt": "Define democracy.", "label": "factual_knowledge"},
    {"prompt": "What is the boiling point of water at sea level?", "label": "factual_knowledge"},
    {"prompt": "Explain what an ecosystem is.", "label": "factual_knowledge"},
    {"prompt": "What is the difference between mass and weight?", "label": "factual_knowledge"},
    {"prompt": "How does a refrigerator keep food cold?", "label": "factual_knowledge"},
    {"prompt": "What is the significance of the Magna Carta?", "label": "factual_knowledge"},
    {"prompt": "Explain what compound interest means.", "label": "factual_knowledge"},
    {"prompt": "What is the water cycle?", "label": "factual_knowledge"},
    {"prompt": "Explain what an API is.", "label": "factual_knowledge"},
    {"prompt": "What is the difference between a comet and an asteroid?", "label": "factual_knowledge"},
    {"prompt": "Formulate a detailed, non-mathematical conceptual explanation of the Byzantine Fault Tolerance consensus mechanism in a decentralized architecture.", "label": "factual_knowledge"}, # newly added complex task by me
    {"prompt": "Elucidate the subtle functional distinctions between memory-mapped I/O and isolated port-mapped I/O at the hardware bus architecture level.", "label": "factual_knowledge"}, # newly added complex task by me
    {"prompt": "Appraise the deep structural mechanisms behind the phenomenon of quantum entanglement and how it relates to the no-cloning theorem.", "label": "factual_knowledge"}, # newly added complex task by me

    # =========================================================================
    # mathematical_reasoning
    # =========================================================================
    {"prompt": "If a shirt costs $40 and is on sale for 25% off, what is the sale price?", "label": "mathematical_reasoning"},
    {"prompt": "A train travels 240 miles in 4 hours. What is its average speed?", "label": "mathematical_reasoning"},
    {"prompt": "If I invest $1000 at 5% annual interest compounded yearly, how much will I have after 3 years?", "label": "mathematical_reasoning"},
    {"prompt": "A recipe serves 4 people and needs 2 cups of flour. How much flour is needed for 10 people?", "label": "mathematical_reasoning"},
    {"prompt": "What is 15% of 340?", "label": "mathematical_reasoning"},
    {"prompt": "A rectangular garden is 12 meters by 8 meters. What is its area and perimeter?", "label": "mathematical_reasoning"},
    {"prompt": "If a car's value depreciates 10% per year, what is a $25000 car worth after 3 years?", "label": "mathematical_reasoning"},
    {"prompt": "Two trains leave stations 300 miles apart heading toward each other at 60 and 90 mph. When do they meet?", "label": "mathematical_reasoning"},
    {"prompt": "A store marks up an item by 40% then offers a 20% discount. What is the net change from original price?", "label": "mathematical_reasoning"},
    {"prompt": "If 3 workers can build a wall in 12 days, how long would it take 4 workers at the same rate?", "label": "mathematical_reasoning"},
    {"prompt": "Calculate the compound annual growth rate if revenue went from $2M to $5M over 4 years.", "label": "mathematical_reasoning"},
    {"prompt": "A population of bacteria doubles every 3 hours. Starting with 500, how many after 12 hours?", "label": "mathematical_reasoning"},
    {"prompt": "What is the sum of the first 50 positive integers?", "label": "mathematical_reasoning"},
    {"prompt": "If a pizza is cut into 8 slices and 3 people each eat 2 slices, what fraction remains?", "label": "mathematical_reasoning"},
    {"prompt": "A car gets 32 miles per gallon. How many gallons are needed for a 720 mile trip?", "label": "mathematical_reasoning"},
    {"prompt": "Project next year's revenue if this year is $450,000 and growth rate is 12% annually.", "label": "mathematical_reasoning"},
    {"prompt": "What is the area of a circle with radius 7cm?", "label": "mathematical_reasoning"},
    {"prompt": "If 60% of a class of 30 students passed, how many students failed?", "label": "mathematical_reasoning"},
    {"prompt": "A loan of $5000 has 8% simple annual interest. What is owed after 2 years?", "label": "mathematical_reasoning"},
    {"prompt": "Solve for x: 3x + 7 = 22", "label": "mathematical_reasoning"},
    {"prompt": "What is 25% of 180?", "label": "mathematical_reasoning"},
    {"prompt": "What is 30% of 250?", "label": "mathematical_reasoning"},
    {"prompt": "What is 8% of 900?", "label": "mathematical_reasoning"},
    {"prompt": "What is 12 divided by 4, times 3?", "label": "mathematical_reasoning"},
    {"prompt": "Formulate a multi-variable geometric projection to determine the exact volume intersecting a standard torus and a regular cylinder.", "label": "mathematical_reasoning"}, # newly added complex task by me
    {"prompt": "A factory outputs components with a defective rate following a Poisson distribution with lambda equal to 2.3 per batch. Devise the exact probability that three consecutive batches contain fewer than two defects each.", "label": "mathematical_reasoning"}, # newly added complex task by me
    {"prompt": "Calculate the maximum throughput capacity of a data pipe utilizing a 128-QAM modulation scheme over a noisy channel with a signal-to-noise ratio of 35dB using Shannon's capacity formula.", "label": "mathematical_reasoning"}, # newly added complex task by me

    # =========================================================================
    # sentiment_classification
    # =========================================================================
    {"prompt": "Classify the sentiment: 'This product completely exceeded my expectations, I love it!'", "label": "sentiment_classification"},
    {"prompt": "What is the sentiment of this review: 'Terrible service, I waited an hour and the food was cold.'", "label": "sentiment_classification"},
    {"prompt": "Is this tweet positive, negative, or neutral: 'The meeting has been rescheduled to 3pm.'", "label": "sentiment_classification"},
    {"prompt": "Classify the sentiment and explain: 'I'm not sure how I feel about the new update, it has some good and bad changes.'", "label": "sentiment_classification"},
    {"prompt": "Label the sentiment of this comment: 'Worst purchase I have ever made, complete waste of money.'", "label": "sentiment_classification"},
    {"prompt": "Determine the sentiment: 'The customer support team resolved my issue quickly and were very polite.'", "label": "sentiment_classification"},
    {"prompt": "Classify this feedback as positive or negative: 'It works fine I guess, nothing special.'", "label": "sentiment_classification"},
    {"prompt": "What sentiment does this express: 'I cannot believe how amazing this concert was tonight!'", "label": "sentiment_classification"},
    {"prompt": "Label the sentiment and justify: 'The instructions were confusing and the app kept crashing.'", "label": "sentiment_classification"},
    {"prompt": "Is this review positive or negative: 'Five stars, would definitely recommend to a friend.'", "label": "sentiment_classification"},
    {"prompt": "Classify the tone of this message: 'Fine. Whatever you say.'", "label": "sentiment_classification"},
    {"prompt": "What is the emotional sentiment here: 'I'm so frustrated, this is the third time it's broken.'", "label": "sentiment_classification"},
    {"prompt": "Determine if this feedback is positive, negative, or mixed: 'Great design but the battery life is disappointing.'", "label": "sentiment_classification"},
    {"prompt": "Label the sentiment of this customer complaint email.", "label": "sentiment_classification"},
    {"prompt": "Classify this restaurant review sentiment: 'Absolutely delicious food, will be back again soon.'", "label": "sentiment_classification"},
    {"prompt": "Classify the sentiment: 'Oh great, another structural update that completely changes the keyboard layouts. Truly a masterclass in driving away long-time developers.'", "label": "sentiment_classification"}, # newly added complex task by me
    {"prompt": "Determine if the implicit intent is hostile or constructive: 'The code works flawlessly under perfect conditions, which unfortunately means it will completely melt our production clusters the second actual users hit it. Brilliant design choices all around.'", "label": "sentiment_classification"}, # newly added complex task by me
    {"prompt": "Label the sentiment: 'I can't say I entirely detest this refactored user interface, though calling it a monumental achievement would be an insult to the word monumental.'", "label": "sentiment_classification"}, # newly added complex task by me

    # =========================================================================
    # text_summarization
    # =========================================================================
    {"prompt": "Summarize the following text in one sentence: [long article about renewable energy policy]", "label": "text_summarization"},
    {"prompt": "Condense this report into 3 bullet points.", "label": "text_summarization"},
    {"prompt": "Provide a summary of this passage in under 50 words.", "label": "text_summarization"},
    {"prompt": "Summarize the main argument of this essay in two sentences.", "label": "text_summarization"},
    {"prompt": "Give me a one-paragraph summary of this meeting transcript.", "label": "text_summarization"},
    {"prompt": "Condense this news article down to a single headline-style sentence.", "label": "text_summarization"},
    {"prompt": "Summarize the key findings of this research paper in 100 words or less.", "label": "text_summarization"},
    {"prompt": "Provide a TL;DR of this document.", "label": "text_summarization"},
    {"prompt": "Summarize this email thread into 3 short bullet points for my manager.", "label": "text_summarization"},
    {"prompt": "Condense the following legal text into plain-language summary under 80 words.", "label": "text_summarization"},
    {"prompt": "Summarize this product description in exactly one sentence.", "label": "text_summarization"},
    {"prompt": "Give a brief summary of this book chapter, no more than 5 sentences.", "label": "text_summarization"},
    {"prompt": "Summarize the plot of this story in a short paragraph.", "label": "text_summarization"},
    {"prompt": "Condense this technical specification into a short executive summary.", "label": "text_summarization"},
    {"prompt": "Condense this comprehensive multi-region network routing log output down to an executive summary listing the exact hardware fault sequence: [dense syslog dump containing hundreds of connection status lines]", "label": "text_summarization"}, # newly added complex task by me
    {"prompt": "Summarize the core structural liabilities outlined across sections 4 through 9 of this multi-tenant commercial indemnity clause into two distinct plain-language risk vectors.", "label": "text_summarization"}, # newly added complex task by me
    {"prompt": "Provide a hyper-concise 20-word TL;DR of this dense technical specification covering zero-knowledge proof verification steps in a decentralized cluster.", "label": "text_summarization"}, # newly added complex task by me

    # =========================================================================
    # named_entity_recognition
    # =========================================================================
    {"prompt": "Extract all named entities from this text: 'Tim Cook announced Apple's new headquarters in Cupertino on Monday.'", "label": "named_entity_recognition"},
    {"prompt": "Identify and label all people, organizations, and locations in this paragraph.", "label": "named_entity_recognition"},
    {"prompt": "Extract all dates and locations mentioned in this news article.", "label": "named_entity_recognition"},
    {"prompt": "List every organization name mentioned in this press release.", "label": "named_entity_recognition"},
    {"prompt": "Identify the person names in this text and their associated roles.", "label": "named_entity_recognition"},
    {"prompt": "Extract entities (person, org, location, date) from: 'Sundar Pichai spoke at Google I/O in Mountain View on May 14, 2024.'", "label": "named_entity_recognition"},
    {"prompt": "Find and label all company names in this financial report.", "label": "named_entity_recognition"},
    {"prompt": "Extract all geographic locations mentioned in this travel itinerary.", "label": "named_entity_recognition"},
    {"prompt": "Identify all named entities in this court transcript and categorize them.", "label": "named_entity_recognition"},
    {"prompt": "List all dates mentioned in this contract along with their context.", "label": "named_entity_recognition"},
    {"prompt": "Extract the person, organization, and location entities from this biography excerpt.", "label": "named_entity_recognition"},
    {"prompt": "Tag every proper noun in this paragraph with its entity type.", "label": "named_entity_recognition"},
    {"prompt": "Extract all nested entity blocks (Person, Organization, Location, Geo-Political Entity, Currency) from this diplomatic cable: 'Ambassador Hans-Dieter of the G7 task force met with CEO Marcus Vance of ApexCorp at the Geneva summit to finalize the 500 million Euro cross-border liquidity facility.'", "label": "named_entity_recognition"}, # newly added complex task by me
    {"prompt": "Tag every highly domain-specific nomenclature block with its category (Protein, Gene, Organism, ChemicalCompound): 'In vitro assays showed that the human TP53 tumor suppressor gene actively inhibits the replication of the SV40 polyomavirus when bound to resveratrol molecules.'", "label": "named_entity_recognition"}, # newly added complex task by me
    {"prompt": "Identify all distinct entities including date ranges and corporate legal structures: 'The Delaware-registered partnership AlphaVance LLC executed a forward merger into Tokyo-based Shinwa Electronics Co. between fiscal Q3 2024 and mid-January 2025.'", "label": "named_entity_recognition"}, # newly added complex task by me

    # =========================================================================
    # code_debugging
    # =========================================================================
    {"prompt": "Find the bug in this Python function and fix it:\ndef add(a, b):\n    return a - b", "label": "code_debugging"},
    {"prompt": "This JavaScript code throws an error, identify and fix the issue:\nfunction sum(arr) { return arr.reduce((a,b) => a + b) }", "label": "code_debugging"},
    {"prompt": "Debug this Python code that's supposed to reverse a string but returns the original.", "label": "code_debugging"},
    {"prompt": "Why does this loop never terminate? Fix the bug.\nwhile i < 10:\n    print(i)", "label": "code_debugging"},
    {"prompt": "This SQL query returns duplicate rows, identify the bug and provide a corrected version.", "label": "code_debugging"},
    {"prompt": "Find and fix the off-by-one error in this array indexing code.", "label": "code_debugging"},
    {"prompt": "This React component re-renders infinitely, identify the bug and fix it.", "label": "code_debugging"},
    {"prompt": "Debug this recursive function that causes a stack overflow.", "label": "code_debugging"},
    {"prompt": "This Python code is supposed to check if a number is prime but always returns True. Find the bug.", "label": "code_debugging"},
    {"prompt": "Identify the null pointer exception risk in this Java code and provide a fix.", "label": "code_debugging"},
    {"prompt": "This function has a race condition, identify it and suggest a corrected implementation.", "label": "code_debugging"},
    {"prompt": "Debug why this API call always returns undefined instead of the expected data.", "label": "code_debugging"},
    {"prompt": "Find the logic error in this sorting algorithm implementation.", "label": "code_debugging"},
    {"prompt": "Identify the structural race condition and deadlocking vulnerabilities in this asynchronous thread pool coordinator, and provide a corrected implementation using non-blocking atomic flags.", "label": "code_debugging"}, # newly added complex task by me
    {"prompt": "Debug why this highly complex Python memory leak persists even after garbage collection sweeps: 'class Node: def __init__(self): self.refs = [] ... cyclic references involving system hooks closures.'", "label": "code_debugging"}, # newly added complex task by me
    {"prompt": "Find the precision drop bottleneck in this optimized fast inverse square root assembly implementation and fix the floating-point bit shift mask.", "label": "code_debugging"}, # newly added complex task by me

    # =========================================================================
    # logical_reasoning
    # =========================================================================
    {"prompt": "Five people sit in a row. Alice is not at either end. Bob is immediately left of Alice. Carol is at one end. Where does everyone sit given all constraints?", "label": "logical_reasoning"},
    {"prompt": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?", "label": "logical_reasoning"},
    {"prompt": "Three boxes are labeled Apples, Oranges, and Mixed, but all labels are wrong. You may pick one fruit from one box to correctly label all boxes. Which box do you pick from and how do you deduce the rest?", "label": "logical_reasoning"},
    {"prompt": "A says 'B is lying.' B says 'C is lying.' C says 'A and B are both lying.' Who is telling the truth?", "label": "logical_reasoning"},
    {"prompt": "Four friends need to cross a bridge at night with one flashlight, taking 1,2,5, and 8 minutes respectively. Only two can cross at once. What is the minimum total time to get everyone across?", "label": "logical_reasoning"},
    {"prompt": "Given the clues: the red house is left of the green house, the owner of the green house drinks coffee, and the person in the middle house drinks milk -- determine the full arrangement.", "label": "logical_reasoning"},
    {"prompt": "If today is not Monday and tomorrow is not Wednesday, what day could today be?", "label": "logical_reasoning"},
    {"prompt": "A farmer needs to cross a river with a wolf, a goat, and a cabbage, but can only take one at a time and cannot leave the wolf with the goat or the goat with the cabbage alone. How does he do it?", "label": "logical_reasoning"},
    {"prompt": "In a group of 5 people, everyone shakes hands with everyone else exactly once. How many handshakes occur, and why?", "label": "logical_reasoning"},
    {"prompt": "Solve this constraint puzzle: seat 4 people around a table such that no two people who dislike each other sit adjacent, given the dislike pairs.", "label": "logical_reasoning"},
    {"prompt": "Eight network routers are arranged in a ring topology. Under a strict load constraint, Router A cannot link to Router C or D. Router E must map directly to Router H. If Router B experiences an isolated failure, deduce the optimal backup sequence satisfying all hop rules.", "label": "logical_reasoning"}, # newly added complex task by me
    {"prompt": "Deduce the true identity of the software bug given these constraint claims: Engineer A states the problem is an OOM loop. Engineer B claims A is mistaken. Engineer C asserts B is completely accurate. If exactly one engineer is telling the truth, which component failed?", "label": "logical_reasoning"}, # newly added complex task by me
    {"prompt": "Solve this strict allocation puzzle: distribute six distinct memory blocks across three independent cache tiers such that no two highly volatile segments sit adjacent, satisfying the provided distance vectors.", "label": "logical_reasoning"}, # newly added complex task by me

    # =========================================================================
    # code_generation
    # =========================================================================
    {"prompt": "Write a Python function that checks if a string is a palindrome.", "label": "code_generation"},
    {"prompt": "Write a function to find the second largest number in a list.", "label": "code_generation"},
    {"prompt": "Write a Python function to merge two sorted lists into one sorted list.", "label": "code_generation"},
    {"prompt": "Write a function that counts the frequency of each word in a given text.", "label": "code_generation"},
    {"prompt": "Write a Python function to check if a given year is a leap year.", "label": "code_generation"},
    {"prompt": "Write a function that flattens a nested list of arbitrary depth.", "label": "code_generation"},
    {"prompt": "Write a Python function to implement binary search on a sorted array.", "label": "code_generation"},
    {"prompt": "Write a function that removes duplicate elements from a list while preserving order.", "label": "code_generation"},
    {"prompt": "Write a Python function to calculate the factorial of a number recursively.", "label": "code_generation"},
    {"prompt": "Write a function that validates whether a given string is a valid email address.", "label": "code_generation"},
    {"prompt": "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes.", "label": "code_generation"},
    {"prompt": "Write a function that converts a given number of seconds into hours, minutes, and seconds.", "label": "code_generation"},
    {"prompt": "Write a Python class implementing a basic stack with push, pop, and peek methods.", "label": "code_generation"},
    {"prompt": "Write a function to determine whether two strings are anagrams of each other.", "label": "code_generation"},
    {"prompt": "Write a production-grade, fully asynchronous message broker client in Go that manages structured TCP connections, handles backpressure streams, and recovers gracefully from cluster partition drops.", "label": "code_generation"}, # newly added complex task by me
    {"prompt": "Implement a completely custom, thread-safe memory allocator in C++ from scratch that allocates fixed-size block metadata pools, optimizes heap fragmentation, and tracks allocations via localized bitmaps.", "label": "code_generation"}, # newly added complex task by me
    {"prompt": "Write a resilient Python decorator pattern from scratch that wraps async HTTP network tasks, executing exponential backoff with random jitter, and triggers a circuit breaker if the failure rate exceeds 15%.", "label": "code_generation"}, # newly added complex task by me
]