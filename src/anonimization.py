from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine


configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {
            "lang_code": "en",
            "model_name": "en_core_web_trf",
        }
    ],
}


provider = NlpEngineProvider(
    nlp_configuration=configuration,
)

nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en"],
)

anonymizer = AnonymizerEngine()


def anonymize_pii_trf(
    text: str,
    language: str = "en",
) -> str:
    results = analyzer.analyze(
        text=text,
        language=language,
    )

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )

    return anonymized.text