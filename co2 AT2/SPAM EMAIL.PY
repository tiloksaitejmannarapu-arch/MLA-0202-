from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
emails = [
    "Congratulations you have won a free prize",
    "You are the lucky winner of a cash reward",
    "Claim your free gift now",
    "Win a brand new smartphone today",
    "You have won a million dollar lottery",
    "Click here to claim your free reward",
    "Congratulations you won a vacation package",
    "Get free money by clicking this link",
    "Urgent claim your prize immediately",
    "You are selected for a special cash bonus",
    "Free offer available for limited time",
    "Win exciting prizes by entering now",
    "Claim your exclusive reward today",
    "You have won a free shopping voucher",
    "Congratulations your lottery ticket has won",
    "Please attend the meeting tomorrow at ten AM",
    "Your project report has been successfully submitted",
    "Please send me the assignment before Monday",
    "The team meeting is scheduled for Friday",
    "Your appointment is confirmed for tomorrow",
    "Please find the project document attached",
    "The class will start at nine AM",
    "Thank you for sending the required information",
    "Please review the report and provide feedback",
    "Your exam schedule has been updated",
    "The meeting room has been changed to room five",
    "Please submit your project files today",
    "Your application has been successfully received",
    "The training session will begin next week",
    "Please contact the office for more information"
]

# 1 = Spam
# 0 = Non-Spam

labels = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
]



total_emails = len(emails)
spam_count = labels.count(1)
nonspam_count = labels.count(0)

print("\n" + "=" * 65)
print("             EMAIL SPAM DETECTION SYSTEM")
print("=" * 65)

print("\nDATASET SUMMARY")
print("-" * 65)
print("Total Emails       :", total_emails)
print("Spam Emails        :", spam_count)
print("Non-Spam Emails    :", nonspam_count)
print("Features           : TF-IDF Word Frequency")
print("Classifier          : Logistic Regression")
print("-" * 65)


X_train, X_test, y_train, y_test = train_test_split(
    emails,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_features, y_train)


y_pred = model.predict(X_test_features)


accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 65)
print("Training Emails    :", len(X_train))
print("Testing Emails     :", len(X_test))
print("Model Accuracy     :", round(accuracy * 100, 2), "%")
print("-" * 65)

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print("-" * 65)
print("                 Predicted")
print("                 Non-Spam    Spam")
print("Actual Non-Spam     ", cm[0][0], "       ", cm[0][1])
print("Actual Spam        ", cm[1][0], "       ", cm[1][1])
print("-" * 65)




feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Get words with highest spam scores
spam_words = sorted(
    zip(coefficients, feature_names),
    reverse=True
)[:10]

print("\nTOP WORDS ASSOCIATED WITH SPAM")
print("-" * 65)

for score, word in spam_words:
    print(f"{word:<25} Score: {score:.3f}")

print("-" * 65)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Email Spam Detection - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Non-Spam", "Spam"])
plt.yticks([0, 1], ["Non-Spam", "Spam"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()


print("\n" + "=" * 65)
print("             NEW EMAIL CLASSIFICATION")
print("=" * 65)

new_email = input("\nEnter a new email message: ")

new_email_features = vectorizer.transform([new_email])

prediction = model.predict(new_email_features)

probability = model.predict_proba(new_email_features)

spam_probability = probability[0][1] * 100
nonspam_probability = probability[0][0] * 100
print("\n" + "-" * 65)
print("EMAIL ANALYSIS RESULT")
print("-" * 65)

print("Email Message:")
print(new_email)

print("\nNon-Spam Probability :", round(nonspam_probability, 2), "%")
print("Spam Probability     :", round(spam_probability, 2), "%")

print("\nFINAL CLASSIFICATION")
print("-" * 65)

if prediction[0] == 1:
    print(">>> SPAM EMAIL")
    print(">>> Action: Move email to Spam/Junk folder")
else:
    print(">>> NON-SPAM EMAIL")
    print(">>> Action: Deliver email to Inbox")

print("-" * 65)
print("              CLASSIFICATION COMPLETED")
print("=" * 65)
