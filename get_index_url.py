import firebase_admin
from firebase_admin import credentials, firestore

try:
    cred = credentials.Certificate('firebase-credentials.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    scans_ref = db.collection('scans').where(filter=firestore.FieldFilter('user_id', '==', 'test_user')).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50)
    docs = scans_ref.stream()
    for doc in docs:
        pass
    print("Query successful!")
except Exception as e:
    with open('error_out.txt', 'w', encoding='utf-8') as f:
        f.write(str(e))
