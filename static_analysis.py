import os
import pandas as pd
import logging
import re
from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import ExternalMethod

# --- Configuration & Constants ---

# Suppress noisy logging from androguard
logging.getLogger("androguard").setLevel(logging.CRITICAL)

# Folder containing APK files to analyze
FOLDER = "downloads"#os.path.join("downloads", "Test")
OUTPUT_FILE = "ics_app_vulnerability_report.csv"
ANDROID_NS = 'http://schemas.android.com/apk/res/android'

# Treat weak ciphers and weak hashes separately if you want.
WEAK_HASHES  = {"MD5", "SHA-1"}                 # keep if you still want to report weak hashes
WEAK_CIPHERS = {"RC4", "DES", "3DES"}           # 3DES = DESede
ECB_REGEX    = re.compile(r"(^|/)ECB($|/)")     # mode detector in Cipher.getInstance

# Methods we care about
CIPHER_GET_INSTANCE = "Ljavax/crypto/Cipher;->getInstance(Ljava/lang/String;)"
MD_GET_INSTANCE     = "Ljava/security/MessageDigest;->getInstance(Ljava/lang/String;)"
RAND_CLASS          = "Ljava/util/Random;"
SECURERAND_SETSEED  = (
    "Ljava/security/SecureRandom;->setSeed([B)V",
    "Ljava/security/SecureRandom;->setSeed(J)V",
    "Ljava/security/SecureRandom;-><init>([B)V",
)

def _method_calls_getinstance_for_cipher(meth):
    return any(CIPHER_GET_INSTANCE in str(t[1]) for t in meth.get_xref_to())

def _method_calls_getinstance_for_md(meth):
    return any(MD_GET_INSTANCE in str(t[1]) for t in meth.get_xref_to())





# Enhanced patterns based on the IVdetector paper
WEAK_CRYPTO_ALGORITHMS = ['MD5', 'SHA-1', 'SHA1', 'RC4', 'DES']
WEAK_CRYPTO_CLASSES = [
    'Ljava/security/MessageDigest;',
    'Ljava/lang/Object;->hashCode',
    'Ljavax/crypto/Cipher;'
]


INSECURE_RNG_CLASSES = [
    'Ljava/util/Random;',
    'Ljava/security/SecureRandom;'
]

# External storage APIs from the paper
EXTERNAL_STORAGE_APIS = [
    'getExternalFilesDir',
    'getExternalCacheDir', 
    'getExternalStorageDirectory',
    'getExternalStoragePublicDirectory'
]

# Internal storage APIs
INTERNAL_STORAGE_APIS = [
    'getFilesDir',
    'getCacheDir',
    'openFileOutput'
]

# File writing APIs
FILE_WRITING_APIS = [
    'Ljava/io/File;-><init>',
    'Landroid/content/Context;->openFileOutput',
    'Ljava/io/FileOutputStream;',
    'Ljava/io/BufferedWriter;'
]

# DoS sensitive APIs from the paper
SENSITIVE_APIS_DOS = [
    'Ljava/lang/System;->exit',
    'Landroid/os/Process;->killProcess',
    'Landroid/content/Context;->startActivity',
    'Landroid/webkit/WebView;->loadUrl',
    'Landroid/webkit/WebView;->loadData',
    'Landroid/widget/Toast;->makeText'
]

# ICS Protocol detection patterns
ICS_PROTOCOL_PATTERNS = {
    'Modbus_TCP': {
        'libraries': ['modbus', 'jamod', 'jlibmodbus', 'j2mod', 'modbus4j'],
        'ports': ['502'],
        'secure': False
    },
    'OPC_UA': {
        'libraries': ['opcua', 'open62514'],
        'patterns': ['opc.tcp://'],
        'ports': ['4840'],
        'secure': True
    },
    'S7Comm': {
        'ports': ['102'],
        'secure': True
    },
    'Ethernet_IP': {
        'libraries': ['opener', 'etherip'],
        'ports': ['44818'],
        'secure': True
    },
    'Fins_Omron': {
        'ports': ['9600'],
        'secure': True
    }
}

# SSL/TLS vulnerability patterns
SSL_VULN_PATTERNS = [
    'ALLOW_ALL_HOSTNAME_VERIFIER',
    'AllowAllHostnameVerifier',
    'return true',
    'return 1'
]

# --- Enhanced Analysis Functions ---

def check_unauthorized_access(apk, analysis):
    """
    Enhanced checks for unauthorized access vulnerabilities.
    Implements IVdetector §4.1 methodology.
    """
    findings = {}
    print('unauthorized access')
    # Check backup configuration
    try:
        app_tag = apk.get_android_manifest_xml().find('application')
        allow_backup = app_tag.get(f'{{{ANDROID_NS}}}allowBackup', 'true')
        findings['allow_backup_enabled'] = (allow_backup.lower() == 'true')
    except:
        findings['allow_backup_enabled'] = True  # Default is true
    
    # Enhanced external storage detection
    findings['external_storage_risk'] = check_external_storage_usage(apk, analysis)
    
    # Enhanced internal storage with backup risk
    findings['internal_storage_backup_risk'] = (
        findings['allow_backup_enabled'] and 
        check_internal_storage_usage(analysis)
    )
    
    # Enhanced weak cryptography detection
    crypto_findings = check_weak_cryptography(analysis)
    findings.update(crypto_findings)
    print("Done unautho")
    
    return findings

def check_external_storage_usage(apk, analysis):
    """
    Detailed external storage vulnerability check following IVdetector methodology.
    """
    print("external storage")
    # Check permissions first
    has_write_permission = (
        'android.permission.WRITE_EXTERNAL_STORAGE' in apk.get_permissions() or
        'android.permission.MANAGE_EXTERNAL_STORAGE' in apk.get_permissions()
    )
    
    if not has_write_permission:
        return False
    
    # Check for actual API usage
    external_api_usage = False
    file_writing_usage = False
    
    for method in analysis.get_methods():
        method_name = method.get_method().name
        class_name = method.get_method().get_class_name()
        
        # Check for external storage API calls
        for api in EXTERNAL_STORAGE_APIS:
            if api in method_name:
                external_api_usage = True
                break
        
        # Check for file writing operations
        for call in method.get_xref_to():
            if any(file_api in str(call[1]) for file_api in FILE_WRITING_APIS):
                file_writing_usage = True
                break
    
    return external_api_usage and file_writing_usage

def check_internal_storage_usage(analysis):
    """
    Check for internal storage usage that could be vulnerable with backup enabled.
    """
    print("Internal storage")
    for method in analysis.get_methods():
        method_name = method.get_method().name
        
        # Check for internal storage API usage
        for api in INTERNAL_STORAGE_APIS:
            if api in method_name:
                return True
        
        # Check for file writing to internal storage
        for call in method.get_xref_to():
            if any(file_api in str(call[1]) for file_api in FILE_WRITING_APIS):
                return True
    
    return False

# def check_weak_cryptography(analysis):
#     """
#     Safer extraction for:
#       - weak_crypto_algorithms (strings actually used in getInstance calls)
#       - insecure_rng_used (java.util.Random OR SecureRandom with setSeed/seed ctor)
#       - weak_crypto_classes_found (MessageDigest/Cipher only)
#     """
#     print("Weak crypto")
#     findings = {
#         "weak_crypto_algorithms": [],    # e.g., ['DES','RC4','SHA-1','ECB']
#         "insecure_rng_used": False,
#         "weak_crypto_classes_found": []  # keep only real crypto classes
#     }

#     # --- 1) Which methods use getInstance? (so we only consider strings from those)
#     methods_calling_cipher = set()
#     methods_calling_md     = set()
#     for m in analysis.get_methods():
#         if _method_calls_getinstance_for_cipher(m):
#             methods_calling_cipher.add(m)
#         if _method_calls_getinstance_for_md(m):
#             methods_calling_md.add(m)

#     # --- 2) Collect strings referenced *inside* those methods
#     # Each StringAnalysis obj -> where it is used
#     weak_tokens = set()
#     for sa in analysis.get_strings():
#         s = str(sa)
#         s_up = s.upper()

#         # For each reference of this string, check which method uses it

#         for  mref in sa.get_xref_from():
#             if mref in methods_calling_cipher or mref in methods_calling_md:
#                 # Normalize DES / 3DES
#                 if re.search(r"\bDESEDE\b", s_up) or "3DES" in s_up:
#                     weak_tokens.add("3DES")
#                 elif re.search(r"\bDES\b", s_up):
#                     weak_tokens.add("DES")
#                 if "RC4" in s_up:
#                     weak_tokens.add("RC4")
#                 if "MD5" in s_up:
#                     weak_tokens.add("MD5")
#                 if s_up.replace(" ", "") in {"SHA-1", "SHA1"}:
#                     weak_tokens.add("SHA-1")

#                 # Detect ECB mode from full transforms like "AES/ECB/PKCS5Padding"
#                 if ECB_REGEX.search(s):
#                     weak_tokens.add("ECB")


#     # --- 3) RNG: flag java.util.Random; SecureRandom only if predictably seeded
#     rng_insecure = False
#     securerand_suspicious = False

#     for m in analysis.get_methods():
#         for t in m.get_xref_to():
#             callee = str(t[1])

#             # Any use of java.util.Random => insecure
#             if RAND_CLASS in callee:
#                 rng_insecure = True

#             # Heuristic: SecureRandom setSeed/seed-ctor seen => potentially insecure
#             if any(sig in callee for sig in SECURERAND_SETSEED):
#                 securerand_suspicious = True

#     # Finalize findings
#     findings["weak_crypto_algorithms"] = sorted(weak_tokens)
#     findings["insecure_rng_used"] = bool(rng_insecure or securerand_suspicious)

#     # Only actual crypto classes in this column
#     classes = set()
#     for m in analysis.get_methods():
#         for t in m.get_xref_to():
#             callee = str(t[1])
#             if "Ljava/security/MessageDigest;" in callee or "Ljavax/crypto/Cipher;" in callee:
#                 classes.add(callee.split("->")[0])  # keep class part only
#     findings["weak_crypto_classes_found"] = sorted(classes)

#     return findings
import re

# --- patterns & constants
ECB_REGEX = re.compile(r"/ECB(?:/|$)", re.I)
CBC_REGEX = re.compile(r"/CBC(?:/|$)", re.I)
CTR_REGEX = re.compile(r"/CTR(?:/|$)", re.I)
GCM_REGEX = re.compile(r"/GCM(?:/|$)", re.I)
PKCS1_REGEX = re.compile(r"PKCS1PADDING", re.I)
OAEP_REGEX = re.compile(r"OAEP", re.I)

RAND_CLASS = "Ljava/util/Random;"
SECURERAND_SETSEED = (
    "Ljava/security/SecureRandom;->setSeed(",
    "Ljava/security/SecureRandom;-><init>([B)V",
    "Ljava/security/SecureRandom;-><init>(J)V",
)

# Biometric API class markers
FINGERPRINT_CRYPTOOBJECTS = (
    "Landroid/hardware/fingerprint/FingerprintManager$CryptoObject;",
    "Landroidx/core/hardware/fingerprint/FingerprintManagerCompat$CryptoObject;",
)
BIOMETRICPROMPT_CRYPTOOBJECT = "Landroidx/biometric/BiometricPrompt$CryptoObject;"

# Android Keystore / key params (best-effort heuristics)
KEYSTORE_PARAM_SPEC_BUILDER = "Landroid/security/keystore/KeyGenParameterSpec$Builder;"
AUTH_REQUIRED_SIG = (
    "Landroid/security/keystore/KeyGenParameterSpec$Builder;->setUserAuthenticationRequired(Z)"
)
AUTH_PARAMS_SIG = (
    "Landroid/security/keystore/KeyGenParameterSpec$Builder;->setUserAuthenticationParameters("
)
STRONGBOX_SIG = (
    "Landroid/security/keystore/KeyGenParameterSpec$Builder;->setIsStrongBoxBacked(Z)"
)
INVALIDATE_ON_ENROLL_SIG = (
    "Landroid/security/keystore/KeyGenParameterSpec$Builder;->setInvalidatedByBiometricEnrollment(Z)"
)

# --- helpers
def _method_sig(m):
    """Stable 'Lpkg/Class;->name(desc)' for MethodAnalysis or EncodedMethod."""
    em = m.get_method() if hasattr(m, "get_method") else m
    return f"{em.get_class_name()}->{em.get_name()}{em.get_descriptor()}"

def _method_calls(m, target_prefix: str) -> bool:
    for _, callee, _ in m.get_xref_to():
        if target_prefix in str(callee):
            return True
    return False

def _calls_cipher_getinstance(m) -> bool:
    return _method_calls(m, "Ljavax/crypto/Cipher;->getInstance(Ljava/lang/String;")

def _calls_md_getinstance(m) -> bool:
    return _method_calls(m, "Ljava/security/MessageDigest;->getInstance(Ljava/lang/String;")

def _calls_mac_getinstance(m) -> bool:
    return _method_calls(m, "Ljavax/crypto/Mac;->getInstance(Ljava/lang/String;")

def _looks_like_rsa_transformation(s_up: str) -> bool:
    # Typical RSA strings start with "RSA/" or equal "RSA/ECB/..."
    return s_up.startswith("RSA/") or s_up == "RSA"

# --- main detector
def check_weak_cryptography(analysis, fallback_broad_scan: bool = True):
    """
    Extracts:
      - weak_crypto_algorithms: tokens found in getInstance(...) contexts only
        (DES/3DES/RC4/MD5/SHA-1 + 'ECB' for symmetric modes; avoids RSA false positives)
      - insecure_rng_used: java.util.Random OR SecureRandom seeded explicitly
      - weak_crypto_classes_found: class names of MessageDigest/Cipher (kept for backward compat)
      - non_aead_modes_found: e.g., AES/CBC or AES/CTR (may be OK if paired with HMAC)
      - aead_modes_found: e.g., AES/GCM
      - rsa_pkcs1_used: True if RSA/…/PKCS1Padding seen (recommend OAEP)
      - hmac_in_use: True if Hmac* via javax.crypto.Mac is observed
      - legacy_biometric_api_used: True if FingerprintManager*CryptoObject used
      - biometricprompt_present: True if androidx.biometric.BiometricPrompt used
      - keystore_* flags: best-effort detection of secure key settings
    """
    findings = {
        "weak_crypto_algorithms": [],
        "insecure_rng_used": False,
        "weak_crypto_classes_found": [],
        "non_aead_modes_found": [],
        "aead_modes_found": [],
        "rsa_pkcs1_used": False,
        "hmac_in_use": False,
        "legacy_biometric_api_used": False,
        "biometricprompt_present": False,
        "keystore_builder_used": False,
        "keystore_auth_required": False,
        "keystore_auth_params_used": False,
        "keystore_strongbox_used": False,
        "keystore_invalidate_on_bio_enroll": False,
    }

    # 1) methods that call getInstance(...)
    cipher_callers, md_callers, mac_callers = set(), set(), set()
    for m in analysis.get_methods():
        if _calls_cipher_getinstance(m):
            cipher_callers.add(_method_sig(m))
        if _calls_md_getinstance(m):
            md_callers.add(_method_sig(m))
        if _calls_mac_getinstance(m):
            mac_callers.add(_method_sig(m))

    # 2) strings used *inside those methods* only
    weak = set()
    non_aead = set()
    aead = set()
    rsa_pkcs1 = False
    hmac_seen = False

    def _maybe_record_from_transformation(s_raw: str):
        nonlocal rsa_pkcs1, hmac_seen
        s = s_raw.strip()
        s_up = s.upper()

        # HMAC (from Mac.getInstance) — not weak; just record presence.
        if s_up.startswith("HMAC"):
            hmac_seen = True

        # RSA treatment: don't flag "ECB" inside RSA; detect PKCS1 vs OAEP.
        if _looks_like_rsa_transformation(s_up):
            if PKCS1_REGEX.search(s_up):
                rsa_pkcs1 = True
            # OAEP is good; nothing to add to 'weak' for RSA/ECB/OAEP.
            return

        # Symmetric cipher modes:
        if GCM_REGEX.search(s):
            aead.add("AES/GCM")
        if CBC_REGEX.search(s):
            non_aead.add("AES/CBC")
        if CTR_REGEX.search(s):
            non_aead.add("AES/CTR")

        # Classical weak algos / modes
        if "3DES" in s_up or re.search(r"\bDESEDE\b", s_up):
            weak.add("3DES")
        elif re.search(r"\bDES\b", s_up):
            weak.add("DES")
        if "RC4" in s_up:
            weak.add("RC4")
        if "MD5" in s_up:
            weak.add("MD5")
        if s_up.replace(" ", "") in {"SHA-1", "SHA1"}:
            weak.add("SHA-1")

        # Only treat ECB as weak for *symmetric* ciphers (not RSA).
        if ECB_REGEX.search(s):
            weak.add("ECB")

    for sa in analysis.get_strings():
        s_val = sa.get_value() if hasattr(sa, "get_value") else str(sa)

        for ref in sa.get_xref_from():  # (class, method, offset/insn)
            ref_method = ref[1]
            sig = _method_sig(ref_method)

            if sig in cipher_callers or sig in md_callers or sig in mac_callers:
                _maybe_record_from_transformation(s_val)

    # Optional fallback if no getInstance callers were seen (some apps inline)
    if not (weak or non_aead or aead or rsa_pkcs1 or hmac_seen) and fallback_broad_scan:
        for sa in analysis.get_strings():
            s_val = sa.get_value() if hasattr(sa, "get_value") else str(sa)
            _maybe_record_from_transformation(s_val)

    # 3) RNG & API heuristics (calls across all methods)
    rng_insecure = False
    sr_suspicious = False
    legacy_bio = False
    biometricprompt = False
    ks_builder = False
    ks_auth_required = False
    ks_auth_params = False
    ks_strongbox = False
    ks_invalidate_on_enroll = False

    for m in analysis.get_methods():
        for _, callee, _ in m.get_xref_to():
            cs = str(callee)

            # RNG checks
            if RAND_CLASS in cs:
                rng_insecure = True
            if any(sig in cs for sig in SECURERAND_SETSEED):
                sr_suspicious = True

            # Biometric API presence
            if any(f in cs for f in FINGERPRINT_CRYPTOOBJECTS):
                legacy_bio = True
            if BIOMETRICPROMPT_CRYPTOOBJECT in cs:
                biometricprompt = True

            # Keystore param builder + secure flags (best-effort)
            if KEYSTORE_PARAM_SPEC_BUILDER in cs:
                ks_builder = True
            if AUTH_REQUIRED_SIG in cs:
                ks_auth_required = True
            if AUTH_PARAMS_SIG in cs:
                ks_auth_params = True
            if STRONGBOX_SIG in cs:
                ks_strongbox = True
            if INVALIDATE_ON_ENROLL_SIG in cs:
                ks_invalidate_on_enroll = True

    # 4) crypto classes present (class part only) — kept for backward compatibility
    classes = set()
    for m in analysis.get_methods():
        for _, callee, _ in m.get_xref_to():
            cs = str(callee)
            if "Ljava/security/MessageDigest;" in cs or "Ljavax/crypto/Cipher;" in cs:
                classes.add(cs.split("->")[0])

    # finalize
    findings["weak_crypto_algorithms"] = sorted(weak)
    findings["insecure_rng_used"] = rng_insecure or sr_suspicious
    findings["weak_crypto_classes_found"] = sorted(classes)

    findings["non_aead_modes_found"] = sorted(non_aead)
    findings["aead_modes_found"] = sorted(aead)
    findings["rsa_pkcs1_used"] = rsa_pkcs1
    findings["hmac_in_use"] = hmac_seen

    findings["legacy_biometric_api_used"] = legacy_bio
    findings["biometricprompt_present"] = biometricprompt

    findings["keystore_builder_used"] = ks_builder
    findings["keystore_auth_required"] = ks_auth_required
    findings["keystore_auth_params_used"] = ks_auth_params
    findings["keystore_strongbox_used"] = ks_strongbox
    findings["keystore_invalidate_on_bio_enroll"] = ks_invalidate_on_enroll

    return findings





# --- main detector
# def check_weak_cryptography(analysis, fallback_broad_scan: bool = True):
#     """
#     Extracts:
#       - weak_crypto_algorithms: tokens found in getInstance(...) contexts only
#       - insecure_rng_used: java.util.Random OR SecureRandom seeded explicitly
#       - weak_crypto_classes_found: class names of MessageDigest/Cipher
#     """
#     findings = {
#         "weak_crypto_algorithms": [],
#         "insecure_rng_used": False,
#         "weak_crypto_classes_found": [],
#     }

#     # 1) methods that call getInstance(...)
#     cipher_callers, md_callers = set(), set()
#     for m in analysis.get_methods():
#         if _calls_cipher_getinstance(m):
#             cipher_callers.add(_method_sig(m))
#         if _calls_md_getinstance(m):
#             md_callers.add(_method_sig(m))

#     # 2) strings used *inside those methods* only
#     weak = set()
#     for sa in analysis.get_strings():
#         s = sa.get_value() if hasattr(sa, "get_value") else str(sa)
#         s_up = s.upper()

#         for ref in sa.get_xref_from():  # (class, method, offset/insn)
#             ref_method = ref[1]
#             sig = _method_sig(ref_method)

#             if sig in cipher_callers or sig in md_callers:
#                 if "3DES" in s_up or re.search(r"\bDESEDE\b", s_up):
#                     weak.add("3DES")
#                 elif re.search(r"\bDES\b", s_up):
#                     weak.add("DES")
#                 if "RC4" in s_up:
#                     weak.add("RC4")
#                 if "MD5" in s_up:
#                     weak.add("MD5")
#                 if s_up.replace(" ", "") in {"SHA-1", "SHA1"}:
#                     weak.add("SHA-1")
#                 if ECB_REGEX.search(s):
#                     weak.add("ECB")

#     # Optional fallback if no getInstance callers were seen (some apps inline)
#     if not weak and fallback_broad_scan:
#         for sa in analysis.get_strings():
#             s = (sa.get_value() if hasattr(sa, "get_value") else str(sa))
#             s_up = s.upper()
#             if "3DES" in s_up or "DESEDE" in s_up:
#                 weak.add("3DES")
#             elif "DES" in s_up:
#                 weak.add("DES")
#             if "RC4" in s_up:
#                 weak.add("RC4")
#             if "MD5" in s_up:
#                 weak.add("MD5")
#             if s_up.replace(" ", "") in {"SHA-1", "SHA1"}:
#                 weak.add("SHA-1")
#             if ECB_REGEX.search(s):
#                 weak.add("ECB")

#     # 3) RNG heuristics
#     rng_insecure = False
#     sr_suspicious = False
#     for m in analysis.get_methods():
#         for _, callee, _ in m.get_xref_to():
#             cs = str(callee)
#             if RAND_CLASS in cs:
#                 rng_insecure = True
#             if any(sig in cs for sig in SECURERAND_SETSEED):
#                 sr_suspicious = True

#     # 4) crypto classes present (class part only)
#     classes = set()
#     for m in analysis.get_methods():
#         for _, callee, _ in m.get_xref_to():
#             cs = str(callee)
#             if "Ljava/security/MessageDigest;" in cs or "Ljavax/crypto/Cipher;" in cs:
#                 classes.add(cs.split("->")[0])

#     findings["weak_crypto_algorithms"] = sorted(weak)
#     findings["insecure_rng_used"] = rng_insecure or sr_suspicious
#     findings["weak_crypto_classes_found"] = sorted(classes)
#     return findings





def check_eavesdropping_and_injection(apk, analysis):
    """
    Enhanced network communication security analysis.
    Implements IVdetector §4.2 methodology.
    """
    findings = {}
    
    # Check cleartext traffic configuration
    try:
        app_tag = apk.get_android_manifest_xml().find('application')
        cleartext_allowed = app_tag.get(f'{{{ANDROID_NS}}}usesCleartextTraffic', 'false')
        findings['cleartext_traffic_permitted'] = (cleartext_allowed.lower() == 'true')
    except:
        findings['cleartext_traffic_permitted'] = False
    
    # Enhanced protocol detection
    protocol_findings = detect_ics_protocols(analysis)
    findings.update(protocol_findings)
    
    # HTTP/HTTPS detection
    http_findings = detect_http_usage(analysis)
    findings.update(http_findings)
    
    # SSL/TLS vulnerability detection
    ssl_findings = detect_ssl_vulnerabilities(analysis)
    findings.update(ssl_findings)
    
    return findings

def detect_ics_protocols(analysis):
    """
    Detect Industrial Control System protocols.
    """
    findings = {
        'ics_protocols_detected': [],
        'insecure_ics_protocols': []
    }
    
    strings = analysis.get_strings()
    class_names = [c.name for c in analysis.get_classes()]
    
    for protocol, config in ICS_PROTOCOL_PATTERNS.items():
        protocol_detected = False
        
        # Check for library usage
        if 'libraries' in config:
            for lib in config['libraries']:
                if any(lib.lower() in name.lower() for name in class_names):
                    protocol_detected = True
                    break
        
        # Check for port numbers or patterns in strings
        if not protocol_detected:
            for string in strings:
                str_val = str(string).lower()
                
                # Check ports
                if 'ports' in config:
                    for port in config['ports']:
                        if port in str_val:
                            protocol_detected = True
                            break
                
                # Check patterns
                if 'patterns' in config:
                    for pattern in config['patterns']:
                        if pattern.lower() in str_val:
                            protocol_detected = True
                            break
                
                if protocol_detected:
                    break
        
        if protocol_detected:
            findings['ics_protocols_detected'].append(protocol)
            if not config['secure']:
                findings['insecure_ics_protocols'].append(protocol)
    
    return findings



def detect_http_usage(analysis):
    """
    Detect HTTP/HTTPS usage with defensible evidence:
      - library presence (OkHttp/Retrofit/HttpURLConnection/etc.)
      - URL literals ('http://', 'https://') extracted from method bodies (no XREF noise)
      - call-site sinks in the same method (connect/execute/enqueue/url/baseUrl/openConnection)
    We mark usage True only when a URL literal and a sink co-occur in the SAME method.
    """

    findings = {
        'http_usage_found': False,
        'https_usage_found': False,
        'http_urls': [],                 # only URLs that co-occur with a sink
        'https_urls': [],                # only URLs that co-occur with a sink
        'http_libraries_detected': [],
        'https_libraries_detected': [],
        'network_sinks_detected': []     # unique sink call-sites seen anywhere
    }

    # Libraries we consider part of an HTTP(S) stack
    HTTP_CLASS_TOKENS = (
        'Ljava/net/HttpURLConnection;',
        'Lorg/apache/http/',             # Apache HttpClient
        'Lokhttp3/',                     # OkHttp 3.x
        'Lcom/squareup/okhttp/',         # OkHttp 2.x
        'Lretrofit2/',                   # Retrofit (uses OkHttp)
        'Lcom/android/volley/',          # Volley
    )
    HTTPS_CLASS_TOKENS = (
        'Ljavax/net/ssl/HttpsURLConnection;',
        'Ljavax/net/ssl/SSLContext;',
        'Ljavax/net/ssl/SSLSocketFactory;',
        'Ljavax/net/ssl/SSLSocket;',
        'Ljavax/net/ssl/TrustManager;',
        'Ljavax/net/ssl/X509TrustManager;',
        'Ljavax/net/ssl/HostnameVerifier;',
        # OkHttp TLS helpers:
        'Lokhttp3/internal/tls/', 'Lokhttp3/TlsVersion', 'Lokhttp3/CertificatePinner',
    )

    # Calls that indicate a real network operation or URL binding
    SINK_TOKENS = (
        # Java/Android core
        'Ljava/net/URL;-><init>(',
        'Ljava/net/URLConnection;->openConnection(',
        'Ljava/net/HttpURLConnection;->connect(',
        'Ljava/net/URLConnection;->getInputStream(',
        'Ljava/io/InputStream;->read(',
        # OkHttp
        'Lokhttp3/OkHttpClient;->newCall(',
        'Lokhttp3/Call;->execute(',
        'Lokhttp3/Call;->enqueue(',
        'Lokhttp3/Request$Builder;->url(',
        # Retrofit
        'Lretrofit2/Retrofit$Builder;->baseUrl(',
        'Lretrofit2/Call;->execute(',
        'Lretrofit2/Call;->enqueue(',
        # Apache HttpClient
        'Lorg/apache/http/client/HttpClient;->execute(',
        'Lorg/apache/http/impl/client/DefaultHttpClient;->execute(',
        # Volley
        'Lcom/android/volley/RequestQueue;->add(',
    )

    # Ignore these constant URLs / namespaces and third-party SDK classes when attributing usage
    NOISE_URLS = {
        'http://schemas.android.com/apk/res/android',
        'http://www.example.com',
        'http://hostname/?',
    }
    NOISE_SUBSTRINGS = ('google-analytics',)
    EXCLUDE_CLASS_PREFIXES = (
        'Lcom/google/', 'Lcom/google/android/gms/', 'Landroid/support/', 'Landroidx/'
    )

    import re
    url_regex = re.compile(r'(https?://[^\s\'"]+)')

    def extract_urls_from_method(m):
        """Return set of http(s) URLs literally referenced in method m (deduped, noise-filtered)."""
        urls = set()
        get_strings = getattr(m, 'get_strings', None)
        if callable(get_strings):
            try:
                for s in get_strings():
                    # Prefer raw literal value if available
                    try:
                        v = s.get_value()
                    except Exception:
                        v = None
                    text = v if isinstance(v, str) else str(s)
                    # Fallback: regex out http(s) substrings even from 'XREFto ...' dumps
                    for u in url_regex.findall(text):
                        if u in NOISE_URLS: 
                            continue
                        if any(noise in u for noise in NOISE_SUBSTRINGS):
                            continue
                        urls.add(u)
            except Exception:
                pass
        return urls

    def method_has_sink(m):
        """True if method m calls any known network sink."""
        try:
            callees = ''.join(str(x[1]) for x in m.get_xref_to())
        except Exception:
            return False
        return any(tok in callees for tok in SINK_TOKENS)

    http_urls, https_urls = set(), set()
    http_libs, https_libs = set(), set()
    sinks = set()

    # 1) Scan methods for (URL literal + sink) evidence; exclude obvious 3P SDK classes for attribution
    for m in analysis.get_methods():
        # Class name (to filter out Google/AndroidX libs when attributing to "the app")
        try:
            cls_name = getattr(m, 'class_name', None) or m.get_method().get_class_name()
        except Exception:
            cls_name = ''
        # Record sinks globally (for reporting)
        try:
            for x in m.get_xref_to():
                callee = str(x[1])
                if any(tok in callee for tok in SINK_TOKENS):
                    sinks.add(callee)
                for lib in HTTP_CLASS_TOKENS:
                    if lib in callee:
                        http_libs.add(lib)
                for lib in HTTPS_CLASS_TOKENS:
                    if lib in callee:
                        https_libs.add(lib)
        except Exception:
            pass

        # Only attribute URL+sink evidence to first-party code (skip obvious 3P SDKs)
        if cls_name.startswith(EXCLUDE_CLASS_PREFIXES):
            continue

        urls_in_m = extract_urls_from_method(m)
        if not urls_in_m:
            continue
        if not method_has_sink(m):
            continue  # literal present but no network call in this method

        for u in urls_in_m:
            if u.startswith('http://'):
                http_urls.add(u)
            elif u.startswith('https://'):
                https_urls.add(u)

    # Populate findings
    findings['http_urls'] = sorted(http_urls)
    findings['https_urls'] = sorted(https_urls)
    findings['http_libraries_detected'] = sorted(http_libs)
    findings['https_libraries_detected'] = sorted(https_libs)
    findings['network_sinks_detected'] = sorted(sinks)

    # 2) Evidence rule: at least one method with URL+sink for each scheme
    findings['http_usage_found'] = bool(http_urls)
    findings['https_usage_found'] = bool(https_urls)

    return findings


def detect_ssl_vulnerabilities(analysis):
    """
    Detect SSL/TLS implementation vulnerabilities.
    """
    findings = {
        'insecure_hostname_verifier': False,
        'insecure_ssl_error_handler': False,
        'ssl_vulnerabilities_found': []
    }
    
    for class_obj in analysis.get_classes():
        class_name = class_obj.name
        
        # Check for hostname verifier issues
        if 'HostnameVerifier' in class_name:
            for method in class_obj.get_methods():
                if method.get_method().name == 'verify':
                    try:
                        source_code = method.get_method().source() if hasattr(method.get_method(), 'source') else ""
                        if source_code:
                            for pattern in SSL_VULN_PATTERNS:
                                if pattern.lower() in source_code.lower():
                                    findings['insecure_hostname_verifier'] = True
                                    findings['ssl_vulnerabilities_found'].append(f"Insecure hostname verifier: {pattern}")
                                    break
                    except:
                        pass
        
        # Check for SSL error handler issues
        if 'WebViewClient' in class_name:
            for method in class_obj.get_methods():
                if 'onReceivedSslError' in method.get_method().name:
                    findings['insecure_ssl_error_handler'] = True
                    findings['ssl_vulnerabilities_found'].append("Insecure SSL error handler found")
    
    return findings

def check_dos_and_service_disruption(apk, analysis):
    """
    Enhanced DoS and service disruption vulnerability analysis.
    Implements IVdetector §4.3 methodology.
    """
    findings = {
        'exported_components_without_permissions': [],
        'exported_components_details': {},
        'sensitive_dos_apis_reachable': []
    }
    
    # Enhanced exported component analysis
    component_analysis = analyze_exported_components(apk)
    findings.update(component_analysis)
    
    # Check for reachable sensitive APIs
    sensitive_api_analysis = check_sensitive_api_reachability(analysis)
    findings.update(sensitive_api_analysis)
    
    return findings






# def detect_http_usage(analysis):
#     """
#     Detect HTTP/HTTPS usage with defensible evidence:
#       - library presence (OkHttp/Retrofit/HttpURLConnection/etc.)
#       - URL strings ('http://', 'https://')
#       - call-site sinks (connect/execute/enqueue/url/baseUrl/openConnection)
#     We mark usage True only when (URL strings) AND (a sink) are both present.
#     """
#     findings = {
#         'http_usage_found': False,
#         'https_usage_found': False,
#         'http_urls': [],
#         'https_urls': [],
#         'http_libraries_detected': [],
#         'https_libraries_detected': [],
#         'network_sinks_detected': []  # call-sites like openConnection/execute/enqueue/url/baseUrl
#     }

#     HTTP_CLASS_TOKENS = (
#         'Ljava/net/HttpURLConnection;',
#         'Lorg/apache/http/',             # Apache HttpClient
#         'Lokhttp3/',                     # OkHttp 3.x
#         'Lcom/squareup/okhttp/',         # OkHttp 2.x
#         'Lretrofit2/',                   # Retrofit (uses OkHttp)
#         'Lcom/android/volley/',          # Volley
#     )
#     HTTPS_CLASS_TOKENS = (
#         'Ljavax/net/ssl/HttpsURLConnection;',
#         'Ljavax/net/ssl/SSLContext;',
#         'Ljavax/net/ssl/SSLSocketFactory;',
#         'Ljavax/net/ssl/SSLSocket;',
#         'Ljavax/net/ssl/TrustManager;',
#         'Ljavax/net/ssl/X509TrustManager;',
#         'Ljavax/net/ssl/HostnameVerifier;',
#         # OkHttp TLS helpers:
#         'Lokhttp3/internal/tls/', 'Lokhttp3/TlsVersion', 'Lokhttp3/CertificatePinner',
#     )

#     # Calls that indicate an actual network operation or URL binding is happening
#     SINK_TOKENS = (
#         # Java/Android core
#         'Ljava/net/URL;-><init>(',
#         'Ljava/net/URLConnection;->openConnection(',
#         'Ljava/net/HttpURLConnection;->connect(',
#         'Ljava/net/URLConnection;->getInputStream(',
#         'Ljava/io/InputStream;->read(',
#         # OkHttp
#         'Lokhttp3/OkHttpClient;->newCall(',
#         'Lokhttp3/Call;->execute(',
#         'Lokhttp3/Call;->enqueue(',
#         'Lokhttp3/Request$Builder;->url(',
#         # Retrofit
#         'Lretrofit2/Retrofit$Builder;->baseUrl(',
#         'Lretrofit2/Call;->execute(',
#         'Lretrofit2/Call;->enqueue(',
#         # Apache HttpClient
#         'Lorg/apache/http/client/HttpClient;->execute(',
#         'Lorg/apache/http/impl/client/DefaultHttpClient;->execute(',
#         # Volley
#         'Lcom/android/volley/RequestQueue;->add(',
#     )

#     http_urls, https_urls = set(), set()
#     http_libs, https_libs = set(), set()
#     sinks = set()

#     # 1) URL strings
#     try:
#         for s in analysis.get_strings():
#             s = str(s)
#             if 'http://' in s:
#                 http_urls.add(s)
#             if 'https://' in s:
#                 https_urls.add(s)
#     except Exception:
#         pass

#     # 2) Libraries + 3) Sinks
#     for m in analysis.get_methods():
#         try:
#             for x in m.get_xref_to():
#                 callee = str(x[1])  # robust string form of the called method
#                 for lib in HTTP_CLASS_TOKENS:
#                     if lib in callee:
#                         http_libs.add(lib)
#                 for lib in HTTPS_CLASS_TOKENS:
#                     if lib in callee:
#                         https_libs.add(lib)
#                 for token in SINK_TOKENS:
#                     if token in callee:
#                         sinks.add(callee)
#         except Exception:
#             continue

#     findings['http_urls'] = sorted(http_urls)
#     findings['https_urls'] = sorted(https_urls)
#     findings['http_libraries_detected'] = sorted(http_libs)
#     findings['https_libraries_detected'] = sorted(https_libs)
#     findings['network_sinks_detected'] = sorted(sinks)

#     # 4) Evidence rule: URL + sink -> usage
#     findings['http_usage_found'] = bool(http_urls and sinks)
#     findings['https_usage_found'] = bool(https_urls and sinks)

#     return findings


# SSL_VULN_PATTERNS = (
#     'ALLOW_ALL_HOSTNAME_VERIFIER',
#     'VERIFY_NONE',
#     'return true',               # HostnameVerifier.verify stub
#     'handler.proceed',           # WebViewClient.onReceivedSslError -> proceed()
# )

# def detect_ssl_vulnerabilities(analysis):
#     """
#     Heuristics for common bad TLS patterns:
#       - Permissive HostnameVerifier.verify(...)
#       - WebViewClient.onReceivedSslError calling handler.proceed(...)
#     """
#     findings = {
#         'insecure_hostname_verifier': False,
#         'insecure_ssl_error_handler': False,
#         'ssl_vulnerabilities_found': []
#     }

#     for cls in analysis.get_classes():
#         cname = getattr(cls, 'name', '')
#         # HostnameVerifier implementations
#         if 'HostnameVerifier' in cname:
#             for m in cls.get_methods():
#                 if getattr(m.get_method(), 'name', '') == 'verify':
#                     src = ''
#                     try:
#                         src = m.get_method().source()
#                     except Exception:
#                         pass
#                     text = (src or str(m))
#                     if any(pat.lower() in text.lower() for pat in SSL_VULN_PATTERNS):
#                         findings['insecure_hostname_verifier'] = True
#                         findings['ssl_vulnerabilities_found'].append(
#                             'Insecure HostnameVerifier (permits all hosts)'
#                         )

#         # WebView SSL error handler
#         if 'WebViewClient' in cname:
#             for m in cls.get_methods():
#                 if 'onReceivedSslError' in getattr(m.get_method(), 'name', ''):
#                     src = ''
#                     try:
#                         src = m.get_method().source()
#                     except Exception:
#                         pass
#                     text = (src or str(m))
#                     if 'proceed' in text:  # handler.proceed(...)
#                         findings['insecure_ssl_error_handler'] = True
#                         findings['ssl_vulnerabilities_found'].append(
#                             'WebView onReceivedSslError allows proceed()'
#                         )

#     return findings


# def analyze_exported_components(apk):
#     """
#     Detailed analysis of exported Android components.
#     """
#     findings = {
#         'exported_components_without_permissions': [],
#         'exported_components_details': {
#             'activities': [],
#             'services': [],
#             'receivers': [],
#             'providers': []
#         }
#     }
    
#     try:
#         manifest_root = apk.get_android_manifest_xml()
#         app_element = manifest_root.find('application')
#         if app_element is None:
#             return findings

#         component_mapping = {
#             'activity': 'activities',
#             'service': 'services', 
#             'receiver': 'receivers',
#             'provider': 'providers'
#         }

#         for tag, list_key in component_mapping.items():
#             for element in app_element.findall(tag):
#                 comp_name = element.get(f'{{{ANDROID_NS}}}name')
#                 if not comp_name:
#                     continue

#                 exported_attr = element.get(f'{{{ANDROID_NS}}}exported')
#                 has_intent_filter = element.find('intent-filter') is not None
#                 permission_attr = element.get(f'{{{ANDROID_NS}}}permission')

#                 # Determine if component is exported
#                 is_exported = False
#                 if exported_attr == 'true':
#                     is_exported = True
#                 elif exported_attr is None and has_intent_filter:
#                     is_exported = True
                
#                 if is_exported:
#                     component_info = {
#                         'name': comp_name,
#                         'type': tag,
#                         'has_permission': bool(permission_attr),
#                         'permission': permission_attr,
#                         'has_intent_filter': has_intent_filter
#                     }
                    
#                     findings['exported_components_details'][list_key].append(component_info)
                    
#                     if not permission_attr:
#                         findings['exported_components_without_permissions'].append(f"{tag}:{comp_name}")
    
#     except Exception as e:
#         print(f"Error analyzing exported components: {e}")
    
#     return findings


# Heuristics for noisy "do-nothing" component names
_IGNORE_NAME_REGEXES = [
    re.compile(r"\bmainactivity\b", re.I),
    re.compile(r"\bmain\b", re.I),
    re.compile(r"\bsplash\b", re.I),
    re.compile(r"\bsplashscreen\b", re.I),
    re.compile(r"\blauncher\b", re.I),
    re.compile(r"\bhomeactivity\b", re.I),
]

def _simple_name(name: str) -> str:
    """
    Best-effort simple class name from android:name which may be:
      - 'com.example.MainActivity'
      - '.MainActivity'
      - 'MainActivity'
    """
    n = (name or "").strip()
    if n.startswith("."):
        n = n[1:]
    # Drop package prefix and inner-class suffixes
    n = n.split(".")[-1]
    n = n.split("$")[-1]
    return n

def _is_launcher_only_activity(element, ANDROID_NS: str) -> bool:
    """
    True if ALL intent-filters are MAIN/LAUNCHER (or LEANBACK_LAUNCHER) and nothing else.
    """
    filters = element.findall('intent-filter')
    if not filters:
        return False

    for ife in filters:
        actions = [a.get(f'{{{ANDROID_NS}}}name') for a in ife.findall('action')]
        cats    = [c.get(f'{{{ANDROID_NS}}}name') for c in ife.findall('category')]
        datas   = ife.findall('data')

        # Must contain MAIN + a launcher category
        has_main = 'android.intent.action.MAIN' in actions
        has_launcher = any(c in ('android.intent.category.LAUNCHER',
                                 'android.intent.category.LEANBACK_LAUNCHER') for c in cats)
        if not (has_main and has_launcher):
            return False

        # No extra actions/categories/data beyond launcher usage
        extra_actions = [a for a in actions if a != 'android.intent.action.MAIN']
        extra_cats = [c for c in cats if c not in ('android.intent.category.LAUNCHER',
                                                   'android.intent.category.LEANBACK_LAUNCHER')]
        if datas or extra_actions or extra_cats:
            return False

    return True

def _should_ignore_component(tag: str, comp_name: str, element, ANDROID_NS: str, permission_attr: str) -> bool:
    """
    Skip components that are:
      - explicitly protected by a permission (noise)
      - obviously "do-nothing" by name (MainActivity, Splash, Launcher, etc.)
      - launcher-only activities (MAIN/LAUNCHER and nothing else)
    """
    # If it already has a permission, we don't need to flag it
    if permission_attr:
        return True

    # Name-based heuristics
    sname = _simple_name(comp_name)
    if any(rx.search(sname) for rx in _IGNORE_NAME_REGEXES):
        return True

    # Launcher-only activities
    if tag == 'activity' and _is_launcher_only_activity(element, ANDROID_NS):
        return True

    return False


def analyze_exported_components(apk):
    """
    Detailed analysis of exported Android components.
    """
    findings = {
        'exported_components_without_permissions': [],
        'exported_components_details': {
            'activities': [],
            'services': [],
            'receivers': [],
            'providers': []
        }
    }
    
    try:
        manifest_root = apk.get_android_manifest_xml()
        app_element = manifest_root.find('application')
        if app_element is None:
            return findings

        component_mapping = {
            'activity': 'activities',
            'service': 'services', 
            'receiver': 'receivers',
            'provider': 'providers'
        }

        # NOTE: ANDROID_NS must exist in your module; using it here as in your original code.
        for tag, list_key in component_mapping.items():
            for element in app_element.findall(tag):
                comp_name = element.get(f'{{{ANDROID_NS}}}name')
                if not comp_name:
                    continue

                exported_attr = element.get(f'{{{ANDROID_NS}}}exported')
                has_intent_filter = element.find('intent-filter') is not None
                permission_attr = element.get(f'{{{ANDROID_NS}}}permission')

                # Determine if component is exported
                is_exported = False
                if exported_attr == 'true':
                    is_exported = True
                elif exported_attr is None and has_intent_filter:
                    is_exported = True

                if is_exported:
                    # >>> NEW: ignore noise & permission-protected components
                    if _should_ignore_component(tag, comp_name, element, ANDROID_NS, permission_attr):
                        continue
                    # <<<

                    component_info = {
                        'name': comp_name,
                        'type': tag,
                        'has_permission': bool(permission_attr),
                        'permission': permission_attr,
                        'has_intent_filter': has_intent_filter
                    }
                    
                    findings['exported_components_details'][list_key].append(component_info)
                    
                    if not permission_attr:
                        findings['exported_components_without_permissions'].append(f"{tag}:{comp_name}")
    
    except Exception as e:
        print(f"Error analyzing exported components: {e}")
    
    return findings


def check_sensitive_api_reachability(analysis):
    """
    Check if sensitive APIs are reachable from exported components.
    """
    findings = {
        'sensitive_dos_apis_reachable': [],
        'sensitive_apis_found': []
    }
    
    for method in analysis.get_methods():
        for call in method.get_xref_to():
            call_str = str(call[1])
            
            for sensitive_api in SENSITIVE_APIS_DOS:
                if sensitive_api in call_str:
                    findings['sensitive_apis_found'].append(sensitive_api)
                    # This is a simplified check - in reality, you'd need proper data flow analysis
                    findings['sensitive_dos_apis_reachable'].append(call_str)
    
    findings['sensitive_apis_found'] = list(set(findings['sensitive_apis_found']))
    
    return findings

def check_ui_modification(analysis):
    """
    Enhanced UI modification vulnerability analysis.
    Implements IVdetector §4.4 methodology.
    """
    findings = {}
    
    # WebView analysis
    webview_findings = analyze_webview_usage(analysis)
    findings.update(webview_findings)
    
    # Database file analysis
    database_findings = analyze_database_usage(analysis)
    findings.update(database_findings)
    
    return findings

# def analyze_webview_usage(analysis):
#     """
#     Analyze WebView usage for UI modification vulnerabilities.
#     """
#     findings = {
#         'webview_loads_http': False,
#         'webview_loads_https': False,
#         'webview_urls_found': []
#     }
    
#     strings = analysis.get_strings()
    
#     # Check for WebView loading URLs
#     for string in strings:
#         str_val = str(string).lower()
#         if 'loadurl' in str_val or 'loaddata' in str_val:
#             if 'http://' in str_val:
#                 findings['webview_loads_http'] = True
#                 findings['webview_urls_found'].append(str(string))
#             elif 'https://' in str_val:
#                 findings['webview_loads_https'] = True
#                 findings['webview_urls_found'].append(str(string))
    
#     # Check for WebView class usage
#     for class_obj in analysis.get_classes():
#         if 'WebView' in class_obj.name:
#             for method in class_obj.get_methods():
#                 method_name = method.get_method().name
#                 if 'loadUrl' in method_name or 'loadData' in method_name:
#                     # This indicates WebView usage for UI
#                     break
    
#     return findings

# def analyze_database_usage(analysis):
#     """
#     Analyze database usage that could lead to UI modification.
#     """
#     findings = {
#         'sqlite_database_used': False,
#         'database_in_external_storage': False,
#         'database_classes_found': []
#     }
    
#     database_indicators = [
#         'SQLiteDatabase',
#         'SQLiteOpenHelper', 
#         'ContentProvider',
#         'Room',
#         '.db',
#         'sqlite'
#     ]
    
#     # Check class names
#     for class_obj in analysis.get_classes():
#         class_name = class_obj.name
#         for indicator in database_indicators:
#             if indicator.lower() in class_name.lower():
#                 findings['sqlite_database_used'] = True
#                 findings['database_classes_found'].append(class_name)
    
#     # Check strings for database files
#     strings = analysis.get_strings()
#     for string in strings:
#         str_val = str(string).lower()
#         if '.db' in str_val or 'sqlite' in str_val:
#             findings['sqlite_database_used'] = True
#             # Simple heuristic: if external storage APIs are used with database, 
#             # it might be stored externally
#             if 'external' in str_val:
#                 findings['database_in_external_storage'] = True
    
#     findings['database_classes_found'] = list(set(findings['database_classes_found']))
    
#     return findings


def analyze_webview_usage(analysis):
    """
    Detect WebView loads and risky configs with defensible signals:
      - call-sites to loadUrl/loadData/loadDataWithBaseURL
      - URL scheme hints (http/https/file/content)
      - risky settings: setJavaScriptEnabled(true), setAllowFileAccess(true),
        setAllowUniversalAccessFromFileURLs(true), setMixedContentMode(...)
      - insecure SSL handler in WebViewClient.onReceivedSslError -> proceed()
    """
    findings = {
        'webview_calls': [],                 # list of dicts: {'callee':..., 'arg_hint':..., 'scheme':...}
        'webview_loads_http': False,
        'webview_loads_https': False,
        'risky_settings': [],               # strings of risky settings found
        'insecure_ssl_error_handler': False,
        'webview_urls_found': []            # literal URL strings if any
    }

    # Gather literal URL strings (helpful but not required)
    http_urls, https_urls = set(), set()
    for s in analysis.get_strings():
        sv = str(s)
        if 'http://' in sv:
            http_urls.add(sv)
        if 'https://' in sv:
            https_urls.add(sv)
    findings['webview_urls_found'] = sorted(http_urls | https_urls)

    # Methods/sinks and settings to look for
    LOAD_TOKENS = (
        'Landroid/webkit/WebView;->loadUrl(',
        'Landroid/webkit/WebView;->loadData(',
        'Landroid/webkit/WebView;->loadDataWithBaseURL(',
    )
    RISKY_SETTING_TOKENS = (
        ('Landroid/webkit/WebSettings;->setJavaScriptEnabled(', 'setJavaScriptEnabled(true)'),
        ('Landroid/webkit/WebSettings;->setAllowFileAccess(', 'setAllowFileAccess(true)'),
        ('Landroid/webkit/WebSettings;->setAllowUniversalAccessFromFileURLs(', 'setAllowUniversalAccessFromFileURLs(true)'),
        ('Landroid/webkit/WebSettings;->setMixedContentMode(', 'setMixedContentMode(...)'),
        ('Landroid/webkit/WebSettings;->setDomStorageEnabled(', 'setDomStorageEnabled(true)'),
    )

    # Scan call-sites
    for m in analysis.get_methods():
        try:
            for x in m.get_xref_to():
                callee = str(x[1])
                # Load calls
                if any(tok in callee for tok in LOAD_TOKENS):
                    # Try to guess scheme from nearby string constants in the method
                    arg_hint = ''
                    scheme = None
                    try:
                        for ss in m.get_strings():
                            ssv = str(ss)
                            if 'http://' in ssv and not scheme:
                                scheme, arg_hint = 'http', ssv
                            if 'https://' in ssv and scheme is None:
                                scheme, arg_hint = 'https', ssv
                            if scheme and arg_hint:
                                break
                    except Exception:
                        pass
                    findings['webview_calls'].append({'callee': callee, 'arg_hint': arg_hint, 'scheme': scheme})
                # Risky settings
                for tok, label in RISKY_SETTING_TOKENS:
                    if tok in callee:
                        # Look for "true" literal in the same method when applicable
                        if 'Enabled(' in tok or 'Allow' in tok or 'DomStorage' in tok:
                            body = (getattr(m.get_method(), 'source', lambda: '')() or str(m))
                            if 'true' in body:
                                findings['risky_settings'].append(label)
                        else:
                            findings['risky_settings'].append(label)

                # Insecure WebView SSL error handler
                if 'Landroid/webkit/WebViewClient;->onReceivedSslError(' in callee:
                    body = (getattr(m.get_method(), 'source', lambda: '')() or str(m))
                    if 'proceed' in body:
                        findings['insecure_ssl_error_handler'] = True
        except Exception:
            continue

    # Final booleans: prefer call-site evidence; fall back to literal URLs
    if any(c.get('scheme') == 'http' for c in findings['webview_calls']) or bool(http_urls):
        findings['webview_loads_http'] = True
    if any(c.get('scheme') == 'https' for c in findings['webview_calls']) or bool(https_urls):
        findings['webview_loads_https'] = True

    # Dedup risky settings
    findings['risky_settings'] = sorted(set(findings['risky_settings']))
    return findings
def analyze_database_usage(analysis):
    """
    Detect SQLite/Room usage and whether DB paths land on external storage.
    Evidence rule:
      - DB sink present (open/create/write)
      - AND method also references an external-storage path or API
    """
    findings = {
        'sqlite_database_used': False,
        'database_in_external_storage': False,
        'database_sinks_found': [],       # list of sink signatures
        'database_paths': [],             # literal paths/names seen
        'external_path_indicators': [],   # evidence strings/APIs indicating external storage
    }

    DB_SINKS = (
        'Landroid/content/Context;->openOrCreateDatabase(',
        'Landroid/database/sqlite/SQLiteDatabase;->openOrCreateDatabase(',
        'Landroid/database/sqlite/SQLiteDatabase;->openDatabase(',
        'Landroid/database/sqlite/SQLiteOpenHelper;->getWritableDatabase(',
        'Landroid/database/sqlite/SQLiteOpenHelper;->getReadableDatabase(',
        # Room:
        'Landroidx/room/Room;->databaseBuilder(',
        'Landroid/arch/persistence/room/Room;->databaseBuilder(',
    )
    EXTERNAL_API_TOKENS = (
        'Landroid/os/Environment;->getExternalStorageDirectory(',
        'Landroid/content/Context;->getExternalFilesDir(',
        'Landroid/content/Context;->getExternalCacheDir(',
        'Landroid/os/Environment;->getExternalStoragePublicDirectory(',
    )
    EXTERNAL_PATH_STRINGS = (
        '/sdcard/', 'storage/emulated/0', 'mnt/sdcard', 'Android/data/',
        'external', 'Download/', 'Documents/', 'Pictures/', 'Movies/'
    )

    db_sinks, db_names, ext_api_hits, ext_path_hits = set(), set(), set(), set()

    # Global string sweep for DB names/paths and external indicators
    for s in analysis.get_strings():
        sv = str(s)
        if sv.endswith('.db') or '.sqlite' in sv or sv.lower().endswith('.db3'):
            db_names.add(sv)
        if any(tok in sv for tok in EXTERNAL_PATH_STRINGS):
            ext_path_hits.add(sv)

    # Method-level evidence: DB sinks and external APIs in same method (cheap x-ref)
    for m in analysis.get_methods():
        try:
            callee_set = set(str(x[1]) for x in m.get_xref_to())
        except Exception:
            callee_set = set()

        if any(sink in ''.join(callee_set) for sink in DB_SINKS):
            findings['sqlite_database_used'] = True
            db_sinks.update([c for c in callee_set if any(s in c for s in DB_SINKS)])

            # Collect method-local strings (potential DB file names or absolute paths)
            try:
                for ss in m.get_strings():
                    ssv = str(ss)
                    if ssv.endswith('.db') or '.sqlite' in ssv:
                        db_names.add(ssv)
                    if any(tok in ssv for tok in EXTERNAL_PATH_STRINGS):
                        ext_path_hits.add(ssv)
            except Exception:
                pass

            # If the SAME method also hits an external storage API, flag external DB
            if any(tok in ''.join(callee_set) for tok in EXTERNAL_API_TOKENS):
                ext_api_hits.update([c for c in callee_set if any(t in c for t in EXTERNAL_API_TOKENS)])

    findings['database_sinks_found'] = sorted(db_sinks)
    findings['database_paths'] = sorted(db_names)
    findings['external_path_indicators'] = sorted(ext_api_hits | set(ext_path_hits))

    # Evidence rule for external DB:
    # (DB sink present) AND (external API or external-looking path seen)
    findings['database_in_external_storage'] = bool(findings['sqlite_database_used'] and findings['external_path_indicators'])

    return findings

    
def check_additional_vulnerabilities(apk, analysis):
    """
    Check for additional security vulnerabilities.
    Implements IVdetector §5.6 additional measurements.
    """
    findings = {}
    
    # APK signature scheme analysis
    findings.update(check_signature_scheme(apk))
    
    # KeyStore usage analysis
    findings.update(check_keystore_usage(analysis))
    
    return findings

def check_signature_scheme(apk):
    """
    Check APK signature scheme vulnerabilities.
    """
    findings = {}
    
    try:
        is_v1_signed = apk.is_signed_v1()
        is_v2_signed = apk.is_signed_v2()
        is_v3_signed = hasattr(apk, 'is_signed_v3') and apk.is_signed_v3()
        
        findings['signed_with_v1_only'] = is_v1_signed and not is_v2_signed and not is_v3_signed
        findings['signature_schemes'] = {
            'v1': is_v1_signed,
            'v2': is_v2_signed, 
            'v3': is_v3_signed
        }
    except Exception as e:
        findings['signed_with_v1_only'] = False
        findings['signature_schemes'] = {'error': str(e)}
    
    return findings

def check_keystore_usage(analysis):
    feats = set()
    saw_keystore = False

    BASELINE = (
        "Ljava/security/KeyStore;",
        "Ljava/security/KeyPairGenerator;",
        "Ljavax/crypto/KeyGenerator;",
    )
    ANDROID_KEYSTORE_STRING = "AndroidKeyStore"
    USER_AUTH_METHODS = (
        "Landroid/security/keystore/KeyGenParameterSpec$Builder;->setUserAuthenticationRequired(Z)V",
        "Landroid/security/keystore/KeyGenParameterSpec$Builder;->setUserAuthenticationParameters(II)V",
        "Landroid/security/keystore/KeyGenParameterSpec$Builder;->setUserAuthenticationValidityDurationSeconds(I)V",
    )
    STRONGBOX_METHOD = "Landroid/security/keystore/KeyGenParameterSpec$Builder;->setIsStrongBoxBacked(Z)V"
    SECURE_ELEMENT_PKG = "Landroid/se/omapi/"  # OMAPI

    # 1) Any baseline KeyStore usage?
    for m in analysis.get_methods():
        for t in m.get_xref_to():
            callee = str(t[1])
            if any(b in callee for b in BASELINE):
                saw_keystore = True
                feats.add("BaselineKeyStore")
            if STRONGBOX_METHOD in callee:
                feats.add("StrongBox")
            if any(u in callee for u in USER_AUTH_METHODS):
                feats.add("UserAuth")
            if SECURE_ELEMENT_PKG in callee:
                feats.add("SecureElement")

    # 2) Did we see the AndroidKeyStore provider string? (TEE-backed)
    for s in analysis.get_strings():
        if str(s) == ANDROID_KEYSTORE_STRING:
            feats.add("AndroidKeyStore")
            saw_keystore = True

    return {
        "keystore_usage_found": saw_keystore,
        "secure_keystore_features": sorted(feats),
    }


def analyze_apk_file(apk_path):
    """
    Main analysis orchestrator for a single APK file.
    Enhanced to follow IVdetector methodology more closely.
    """
    try:
        print(f"[*] Loading APK: {os.path.basename(apk_path)}")
        apk, _, analysis = AnalyzeAPK(apk_path)
        app_name = apk.get_app_name()
        
        print(f"[*] Analyzing: {app_name}")
        
        results = {'apk_name': app_name}
        
        # Core vulnerability categories from IVdetector paper
        print("  - Checking unauthorized access vulnerabilities...")
        results.update(check_unauthorized_access(apk, analysis))
        
        print("  - Checking eavesdropping and command injection...")
        results.update(check_eavesdropping_and_injection(apk, analysis))
        
        print("  - Checking DoS and service disruption...")
        results.update(check_dos_and_service_disruption(apk, analysis))
        
        print("  - Checking UI modification vulnerabilities...")
        results.update(check_ui_modification(analysis))
        
        print("  - Checking additional vulnerabilities...")
        results.update(check_additional_vulnerabilities(apk, analysis))
        
        # Calculate risk score
        results['risk_score'] = calculate_risk_score(results)
        
        print(f"  ✓ Analysis complete - Risk Score: {results['risk_score']}")
        
        return results

    except Exception as e:
        print(f"[!] Error analyzing {os.path.basename(apk_path)}: {e}")
        return {'apk_name': os.path.basename(apk_path), 'analysis_error': str(e)}

def calculate_risk_score(results):
    """
    Calculate a risk score based on found vulnerabilities.
    """
    score = 0
    
    # Unauthorized access (high risk)
    if results.get('external_storage_risk', False):
        score += 3
    if results.get('internal_storage_backup_risk', False):
        score += 2
    if results.get('weak_crypto_algorithms', []):
        score += len(results['weak_crypto_algorithms'])
    
    # Network vulnerabilities (high risk)
    if results.get('cleartext_traffic_permitted', False):
        score += 3
    if results.get('insecure_ics_protocols', []):
        score += len(results['insecure_ics_protocols']) * 2
    if results.get('insecure_hostname_verifier', False):
        score += 2
    
    # DoS vulnerabilities (medium risk)
    if results.get('exported_components_without_permissions', []):
        score += len(results['exported_components_without_permissions']) * 0.5
    
    # Additional vulnerabilities
    if results.get('signed_with_v1_only', False):
        score += 1
    
    return round(score, 1)

# --- Main Execution ---

def main():
    """
    Enhanced main function with better reporting.
    """
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Created directory '{FOLDER}'. Please place your APK files inside it and run the script again.")
        return
    
    apk_files = [os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if f.endswith('.apk')]
    
    if not apk_files:
        print(f"No APK files found in the '{FOLDER}' directory.")
        return
    
    print(f"[+] Found {len(apk_files)} APK files to analyze")
    print("=" * 60)
    
    all_results = []
    successful_analyses = 0
    
    for i, apk_path in enumerate(apk_files, 1):
        print(f"\n[{i}/{len(apk_files)}] Processing: {os.path.basename(apk_path)}")
        result = analyze_apk_file(apk_path)
        if result:
            all_results.append(result)
            if 'analysis_error' not in result:
                successful_analyses += 1
    
    if all_results:
        # Create detailed report
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "=" * 60)
        print(f"[+] Analysis complete!")
        print(f"[+] Successfully analyzed: {successful_analyses}/{len(apk_files)} apps")
        print(f"[+] Detailed report saved to: '{OUTPUT_FILE}'")
        
        # Print summary statistics
        if successful_analyses > 0:
            print(f"\n[+] Vulnerability Summary:")
            
            # Count apps with each vulnerability type
            external_storage = sum(1 for r in all_results if r.get('external_storage_risk', False))
            backup_risk = sum(1 for r in all_results if r.get('internal_storage_backup_risk', False))
            weak_crypto = sum(1 for r in all_results if r.get('weak_crypto_algorithms', []))
            cleartext = sum(1 for r in all_results if r.get('cleartext_traffic_permitted', False))
            exported_components = sum(1 for r in all_results if r.get('exported_components_without_permissions', []))
            v1_only = sum(1 for r in all_results if r.get('signed_with_v1_only', False))
            
            print(f"  - External storage risk: {external_storage}/{successful_analyses} ({external_storage/successful_analyses*100:.1f}%)")
            print(f"  - Internal storage backup risk: {backup_risk}/{successful_analyses} ({backup_risk/successful_analyses*100:.1f}%)")
            print(f"  - Weak cryptography: {weak_crypto}/{successful_analyses} ({weak_crypto/successful_analyses*100:.1f}%)")
            print(f"  - Cleartext traffic: {cleartext}/{successful_analyses} ({cleartext/successful_analyses*100:.1f}%)")
            print(f"  - Unprotected exported components: {exported_components}/{successful_analyses} ({exported_components/successful_analyses*100:.1f}%)")
            print(f"  - V1 signature only: {v1_only}/{successful_analyses} ({v1_only/successful_analyses*100:.1f}%)")
            
            # Average risk score
            risk_scores = [r.get('risk_score', 0) for r in all_results if 'analysis_error' not in r]
            if risk_scores:
                avg_risk = sum(risk_scores) / len(risk_scores)
                print(f"  - Average risk score: {avg_risk:.1f}")
        
    else:
        print("\n[!] No applications were successfully analyzed.")

if __name__ == "__main__":
    main()





# def check_weak_cryptography(analysis):
#     """
#     Safer extraction for:
#       - weak_crypto_algorithms (strings actually used in getInstance calls)
#       - insecure_rng_used (java.util.Random OR SecureRandom with setSeed/seed ctor)
#       - weak_crypto_classes_found (MessageDigest/Cipher only)
#     """
#     print("Weak crypto")
#     findings = {
#         "weak_crypto_algorithms": [],    # e.g., ['DES','RC4','SHA-1','ECB']
#         "insecure_rng_used": False,
#         "weak_crypto_classes_found": []  # keep only real crypto classes
#     }

#     # --- 1) Which methods use getInstance? (so we only consider strings from those)
#     methods_calling_cipher = set()
#     methods_calling_md     = set()
#     for m in analysis.get_methods():
#         if _method_calls_getinstance_for_cipher(m):
#             methods_calling_cipher.add(m)
#         if _method_calls_getinstance_for_md(m):
#             methods_calling_md.add(m)

#     # --- 2) Collect strings referenced *inside* those methods
#     # Each StringAnalysis obj -> where it is used
#     weak_tokens = set()
#     for sa in analysis.get_strings():
#         s = str(sa)
#         s_up = s.upper()

#         # For each reference of this string, check which method uses it

#         for  mref in sa.get_xref_from():
#             if mref in methods_calling_cipher or mref in methods_calling_md:
#                 # Normalize DES / 3DES
#                 if re.search(r"\bDESEDE\b", s_up) or "3DES" in s_up:
#                     weak_tokens.add("3DES")
#                 elif re.search(r"\bDES\b", s_up):
#                     weak_tokens.add("DES")
#                 if "RC4" in s_up:
#                     weak_tokens.add("RC4")
#                 if "MD5" in s_up:
#                     weak_tokens.add("MD5")
#                 if s_up.replace(" ", "") in {"SHA-1", "SHA1"}:
#                     weak_tokens.add("SHA-1")

#                 # Detect ECB mode from full transforms like "AES/ECB/PKCS5Padding"
#                 if ECB_REGEX.search(s):
#                     weak_tokens.add("ECB")


#     # --- 3) RNG: flag java.util.Random; SecureRandom only if predictably seeded
#     rng_insecure = False
#     securerand_suspicious = False

#     for m in analysis.get_methods():
#         for t in m.get_xref_to():
#             callee = str(t[1])

#             # Any use of java.util.Random => insecure
#             if RAND_CLASS in callee:
#                 rng_insecure = True

#             # Heuristic: SecureRandom setSeed/seed-ctor seen => potentially insecure
#             if any(sig in callee for sig in SECURERAND_SETSEED):
#                 securerand_suspicious = True

#     # Finalize findings
#     findings["weak_crypto_algorithms"] = sorted(weak_tokens)
#     findings["insecure_rng_used"] = bool(rng_insecure or securerand_suspicious)

#     # Only actual crypto classes in this column
#     classes = set()
#     for m in analysis.get_methods():
#         for t in m.get_xref_to():
#             callee = str(t[1])
#             if "Ljava/security/MessageDigest;" in callee or "Ljavax/crypto/Cipher;" in callee:
#                 classes.add(callee.split("->")[0])  # keep class part only
#     findings["weak_crypto_classes_found"] = sorted(classes)

#     return findings
