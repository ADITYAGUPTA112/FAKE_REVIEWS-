import sys
sys.path.append('.')
from analyzer import predict_review

test_reviews = [
    "This product is terrible! Fake and broke after one day.",
    "I absolutely love this product it is amazing",
    "I was paid to write this review.",
]

for r in test_reviews:
    print(r)
    print(predict_review(r))
