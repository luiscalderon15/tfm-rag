# Regular expression patterns for PII detection

EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PHONE_REGEX = r"\b(\+?\d[\d\s\-]{7,}\d)\b"
PASSPORT_REGEX = r"\b[A-Z0-9]{7,9}\b"
AGE_REGEX = r"\b\d{1,2}\s?(years?|yrs?)\s?(old)?\b"