from defenses import link_allowlist

samples = [
    "Reset your password at https://password.acme.example.",
    "Reset your password at https://acme-login-help.example.",
    "Use acme-payroll-update.example to update your bank details.",
    "A lookalike: https://acme.example.help-portal.com/login",
]
for text in samples:
    cleaned, blocked = link_allowlist(text)
    print("IN: ", text)
    print("OUT:", cleaned.replace("\n", " "))
    print("BLOCKED:", [b["reason"] for b in blocked], "\n")