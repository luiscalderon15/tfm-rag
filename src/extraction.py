from typing import Optional
from pydantic import BaseModel, Field
import ollama
 
# class CV(BaseModel):
#     summary: Optional[str] = Field(
#         None,
#         description=(
#             "The candidate's introductory/self-description text, found under headings such as "
#             "'About me', 'Summary', 'Profile', 'Professional Profile', 'Objective', 'Perfil', "
#             "'Resumen' or similar. If the CV has no such section but starts with a short "
#             "descriptive paragraph before listing experience/education, use that instead. "
#             "None if not present."
#         ),
#     )
#     experience: list[str] = Field(
#         default_factory=list,
#         description=(
#             "All professional/practical background the candidate lists, regardless of how the "
#             "CV organizes it: work experience, employment history, key projects, freelance work, "
#             "internships, volunteering, research work, or academic projects. Covers headings like "
#             "'Experience', 'Work Experience', 'Employment History', 'Projects', 'Key Projects', "
#             "'Experiencia Laboral'. One list entry per distinct job/role/project, each as free "
#             "text with whatever details are available (company or project name, role, dates, "
#             "what was done). Empty list if none found."
#         ),
#     )
 
# MODEL_NAME = "qwen2.5:7b-instruct"
 
# SYSTEM_PROMPT = """You are a precise CV/resume information extractor.
 
# You will be given the raw text of a CV. Extract ONLY the fields defined in the \
# provided JSON schema. Rules:
 
# - Do not invent, infer, or guess information that is not explicitly present in the text.
# - Preserve the original wording of the CV as much as possible instead of paraphrasing.
# - If a field/section is not present in the CV, leave it empty (null or empty list, \
# as appropriate for that field) instead of making something up.
# - The CV may be in Spanish, English, or a mix of both; extract regardless of language.
# - Return ONLY the JSON object matching the schema, nothing else.
# """

# def parse_cv(cv_text: str, model: str = MODEL_NAME) -> CV:
#     response = ollama.chat(
#         model=model,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": cv_text},
#         ],
#         format=CV.model_json_schema(),
#         options={"temperature": 0},
#     )
#     return CV.model_validate_json(response["message"]["content"])


# MODEL_NAME = "qwen2.5:7b-instruct"

# SYSTEM_PROMPT = """You are a precise CV/resume information extractor.
 
# You will be given the raw text of a CV extracted from a PDF. Because PDF text \
# extraction reads column-by-column or block-by-block, the text is often OUT OF ORDER: \
# a section's heading (e.g. "ABOUT ME", "KEY PROJECTS") may appear far from its own \
# content, sometimes even after it. Do not rely on linear/reading order — use semantic \
# understanding to figure out which paragraph belongs to which section.
 
# Extract ONLY the fields defined in the provided JSON schema. Rules:
 
# - Do not invent, infer, or guess information that is not explicitly present in the text.
# - Preserve the original wording of the CV as much as possible instead of paraphrasing.
# - Reassemble scrambled sections: match orphaned paragraphs to the section heading they \
# describe, even if they are not adjacent in the raw text.
# - Rely on each field's description in the schema to know what belongs in it — it \
# already lists the possible section headings/synonyms for that field.
# - "experience" must include EVERY job, internship, and project the candidate lists, \
# merging sections like "WORK EXPERIENCE" and "KEY PROJECTS"/"PROJECTS" into the same list \
# (each project is its own entry, same as each job).
# - If a field/section is not present in the CV, leave it empty (null or empty list, \
# as appropriate for that field) instead of making something up.
# - The CV may be in Spanish, English, or a mix of both; extract regardless of language.
# - Return ONLY the JSON object matching the schema, nothing else.
# """
# # Few-shot example: a real CV with scrambled PDF extraction order, showing how
# # "ABOUT ME" content appears before its own heading, and how KEY PROJECTS must be
# # merged into "experience" alongside WORK EXPERIENCE entries.
# EXAMPLE_INPUT = """DAVID DE LAS HERAS
# PERSONAL SKILLS
# Degree in Mathematics
# Engineering
# Universidad Complutense de Madrid
# 2016 - 2020
# EDUCATION
# Business impact through end-to-end AI and ML solutions, from design to deployment.
# Passionate about solving business challenges through data and AI, with more than 6 \
# years of experience at Accenture delivering advanced analytics and machine learning \
# solutions across industries such as Telco, Financial Services, Resources, and \
# Products. Specialized in the design and implementation of scalable forecasting \
# systems, MLOps architectures, and Generative AI use cases.
# ABOUT ME
# WORK EXPERIENCE
# 2019 - On going
# Data & AI Data Scientist
# Accenture - Strategy & Consulting
# Application of machine learning and analytics techniques to address complex \
# business problems, working closely with stakeholders to align data solutions with \
# strategic goals.
# Data & AI Intern
# 2017 - 2018
# Accenture - Strategy & Consulting
# ZAI Program
# KEY PROJECTS
# Built attribution models to quantify the impact of corporate events and marketing \
# initiatives on revenue and sales performance.
# Attribution Modeling for Business Impact (2019-2020)
# Designed and deployed predictive models such as churn and upsell propensity to \
# generate optimized customer segments for targeted marketing actions.
# Predictive Targeting for Marketing Campaigns (2020-2022)"""

# EXAMPLE_OUTPUT = CV(
#     summary=(
#         "Passionate about solving business challenges through data and AI, with "
#         "more than 6 years of experience at Accenture delivering advanced analytics "
#         "and machine learning solutions across industries such as Telco, Financial "
#         "Services, Resources, and Products. Specialized in the design and "
#         "implementation of scalable forecasting systems, MLOps architectures, and "
#         "Generative AI use cases."
#     ),
#     experience=[
#         '''Data & AI Data Scientist - Accenture - Strategy & Consulting - 2019 - On
#         going: Application of machine learning and analytics techniques to address 
#         complex business problems, working closely with stakeholders to align data 
#         "solutions with strategic goals''',
#         '''Data & AI Intern - Accenture - Strategy & Consulting - ZAI Program - 2017 - 2018
#         Attribution Modeling for Business Impact (2019-2020): Built attribution
#         models to quantify the impact of corporate events and marketing initiatives
#         on revenue and sales performance''',
#         '''Predictive Targeting for Marketing Campaigns (2020-2022): Designed and
#         deployed predictive models such as churn and upsell propensity to generate
#         optimized customer segments for targeted marketing actions.''',
#     ],
# ).model_dump_json()


# def parse_cv(cv_text: str, model: str = MODEL_NAME) -> CV:
#     response = ollama.chat(
#         model=model,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": EXAMPLE_INPUT},
#             {"role": "assistant", "content": EXAMPLE_OUTPUT},
#             {"role": "user", "content": cv_text},
#         ],
#         format=CV.model_json_schema(),
#         options={"temperature": 0},
#     )
#     return CV.model_validate_json(response["message"]["content"])

## Este parecia bien excepto por duplicados
# class CV(BaseModel):
#     summary: Optional[str] = Field(
#         None,
#         description=(
#             "The candidate's introductory/self-description text, found under the name without explicit saying or under headings such as "
#             " 'Their name' 'About me', 'Summary', 'Profile', 'Professional Profile', 'Objective', 'Perfil', "
#             "'Resumen' or similar. If the CV has no such section but starts with a short "
#             "descriptive paragraph before listing experience/education, use that instead. "
#             "None if not present but first check under or behind the name if it is not explicit written."
#         ),
#     )
#     work_experience: list[str] = Field(
#         default_factory=list,
#         description=(
#             "High-level employment history: jobs, internships and roles the candidate has "
#             "held, each usually summarized briefly (company, job title, dates, and a short "
#             "description of responsibilities). Covers headings like 'Work Experience', "
#             "'Experience', 'Employment History', 'Professional Experience', 'Career History', "
#             "'Experiencia Laboral'. One list entry per distinct role/employer, each as free "
#             "text with whatever details are available. Empty list if none found."
#         ),
#     )
#     projects: list[str] = Field(
#         default_factory=list,
#         description=(
#             "Specific projects the candidate describes in detail (what was built, tools used, "
#             "outcome/impact), as opposed to a general job summary. Covers headings like "
#             "'Key Projects', 'Projects', 'Selected Projects', 'Case Studies', 'Portfolio', "
#             "'Proyectos Destacados'. Also include personal/side/academic projects if described "
#             "with this level of detail. One list entry per distinct project, each as free text. "
#             "Empty list if none found."
#         ),
#     )

# class CV(BaseModel):
#     summary: Optional[str] = Field(
#         None,
#         description=(
#             "A short (1-4 sentence) overarching characterization of who the candidate is "
#             "professionally as a whole: their role/title, general area(s) of expertise, "
#             "overall years of experience, and/or an aspirational closing statement (e.g. "
#             "'passionate about...', 'committed to...'). Often found under headings such as "
#             "'About me', 'Summary', 'Profile', 'Professional Profile', 'Objective', 'Perfil', "
#             "'Resumen', but very frequently has NO heading at all. "
#             "Because PDF text extraction can scramble section order, this paragraph may end up "
#             "ANYWHERE in the raw text — not necessarily near the candidate's name or at the top "
#             "— so do NOT rely on its position. Instead identify it by its content: unlike "
#             "work_experience/projects/education entries, it does not mention a specific company, "
#             "project, degree, or date range; it is a generic, high-level statement that could "
#             "describe the person's whole career (e.g. 'Data Scientist with a background in X, "
#             "skilled in Y, passionate about Z'). None if genuinely not present."
#         ),
#     )
#     work_experience: list[str] = Field(
#         default_factory=list,
#         description=(
#             "High-level employment history: jobs, internships and roles the candidate has "
#             "held, each usually summarized briefly (company, job title, dates, and a short "
#             "description of responsibilities)."
#             "Do NOT include the company name or city/location or Dates in the entry."
#             "Covers headings like 'Work Experience', "
#             "'Experience', 'Employment History', 'Professional Experience', 'Career History', "
#             "'Experiencia Laboral'. One list entry per distinct role/employer, each as free "
#             "text with whatever details are available. DO NOT repeat experiences even if dates are different."
#             "Empty list if none found"
#         ),
#     )
#     projects: list[str] = Field(
#         default_factory=list,
#         description=(
#             "Specific projects the candidate describes in detail (what was built, tools used, "
#             "outcome/impact), as opposed to a general job summary. Covers headings like "
#             "'Key Projects', 'Projects', 'Selected Projects', 'Case Studies', 'Portfolio', "
#             "'Proyectos Destacados'. Also include personal/side/academic projects if described "
#             "with this level of detail. One list entry per distinct project, each as free text. "
#             "DO NOT repeat projects even if dates are different."
#             "Empty list if none found."
#         ),
#     )
#     technical_skills: list[str] = Field(
#         default_factory=list,
#         description=(
#             "Technical/hard skills: programming languages, frameworks, libraries, tools, "
#             "software, platforms, databases, cloud services, methodologies (e.g. 'Python', "
#             "'AWS', 'Docker', 'Scrum'). Covers headings like 'Technical Skills', 'Skills', "
#             "'Tech Stack', 'Tools', 'Programming Languages', 'Habilidades Técnicas', "
#             "'Conocimientos'. Do NOT include soft skills (e.g. 'leadership', 'communication'). "
#             "One list entry per skill/tool. Empty list if none found."
#         ),
#     )
#     certifications: list[str] = Field(
#         default_factory=list,
#         description=(
#             "Professional certifications, licenses, or certified courses the candidate holds "
#             "(e.g. 'AWS Certified Solutions Architect', 'PMP'). Covers headings like "
#             "'Certifications', 'Licenses', 'Certificates', 'Certificaciones'. Include issuer "
#             "and/or date if available in the same entry. One list entry per certification. "
#             "Empty list if none found."
#         ),
#     )
#     languages: list[str] = Field(
#         default_factory=list,
#         description=(
#             "Spoken/written languages the candidate knows. One list entry per language, NOT a "
#             "single combined string. If a proficiency level is stated, format the entry as "
#             "'Language: Level' (e.g. 'English: Native', 'Spanish: Professional working "
#             "proficiency'). If no level is stated for a language, just include the language "
#             "name on its own (e.g. 'French'). Never merge multiple languages into one entry "
#             "like 'Spanish, English, Italian' — always split them. Do NOT include contact "
#             "details here (email address, phone number, physical address, LinkedIn/GitHub/"
#             "portfolio URLs, website) even if they appear physically close to the languages "
#             "section in the raw text — those belong to contact info, not languages. Only "
#             "include entries that name an actual language (e.g. 'Spanish', 'English', "
#             "'Mandarin'). Empty list if none found."
#         )
#     )


class CV(BaseModel):
    summary: Optional[str] = Field(
        None,
        description=(
            "A short (1-4 sentence) overarching characterization of who the candidate is "
            "professionally as a whole: their role/title, general area(s) of expertise, "
            "overall years of experience, and/or an aspirational closing statement (e.g. "
            "'passionate about...', 'committed to...'). Often found under headings such as "
            "'About me', 'Summary', 'Profile', 'Professional Profile', 'Objective', 'Perfil', "
            "'Resumen', but very frequently has NO heading at all. "
            "Because PDF text extraction can scramble section order, this paragraph may end up "
            "ANYWHERE in the raw text — not necessarily near the candidate's name or at the top "
            "— so do NOT rely on its position. Instead identify it by its content: unlike "
            "work_experience/projects/education entries, it does not mention a specific company, "
            "project, degree, or date range; it is a generic, high-level statement that could "
            "describe the person's whole career (e.g. 'Data Scientist with a background in X, "
            "skilled in Y, passionate about Z'). None if genuinely not present."
        ),
    )
    work_experience: list[str] = Field(
        default_factory=list,
        description=(
        "Professional roles, jobs and internships held by the candidate. "
        "The main objective is to preserve what the candidate actually did: "
        "responsibilities, tasks, achievements, technical methods, tools used, "
        "problems solved and results obtained. One list entry per distinct role. "
        "Begin each entry with the job title when available, followed by a "
        "detailed free-text description of the candidate's work. "
        "Do NOT include company names, cities, locations or employment dates. "
        "Do NOT split the responsibilities of one role into multiple entries. "
        "Do NOT move normal employment responsibilities into projects. "
        "Do NOT repeat experiences, even if different dates appear. "
        "Empty list if none found."
        ),
    )
    projects: list[str] = Field(
        default_factory=list,
        description=(
            "Specific projects the candidate describes in detail (what was built, tools used, "
            "outcome/impact), as opposed to a general job summary. Covers headings like "
            "'Key Projects', 'Projects', 'Selected Projects', 'Case Studies', 'Portfolio', "
            "'Proyectos Destacados'. Also include personal/side/academic projects if described "
            "with this level of detail. One list entry per distinct project, each as free text. "
            "DO NOT repeat projects even if dates are different."
            "Empty list if none found."
        ),
    )
    technical_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Technical/hard skills: programming languages, frameworks, libraries, tools, "
            "software, platforms, databases, cloud services, methodologies (e.g. 'Python', "
            "'AWS', 'Docker', 'Scrum'). Covers headings like 'Technical Skills', 'Skills', "
            "'Tech Stack', 'Tools', 'Programming Languages', 'Habilidades Técnicas', "
            "'Conocimientos'. Do NOT include soft skills (e.g. 'leadership', 'communication'). "
            "One list entry per skill/tool. Empty list if none found."
        ),
    )
    certifications: list[str] = Field(
        default_factory=list,
        description=(
            "Professional certifications, licenses, or certified courses the candidate holds "
            "(e.g. 'AWS Certified Solutions Architect', 'PMP'). Covers headings like "
            "'Certifications', 'Licenses', 'Certificates', 'Certificaciones'. Include issuer "
            "and/or date if available in the same entry. One list entry per certification. "
            "Empty list if none found."
        ),
    )
    languages: list[str] = Field(
        default_factory=list,
        description=(
            "Spoken/written languages the candidate knows. One list entry per language, NOT a "
            "single combined string. If a proficiency level is stated, format the entry as "
            "'Language: Level' (e.g. 'English: Native', 'Spanish: Professional working "
            "proficiency'). If no level is stated for a language, just include the language "
            "name on its own (e.g. 'French'). Never merge multiple languages into one entry "
            "like 'Spanish, English, Italian' — always split them. Do NOT include contact "
            "details here (email address, phone number, physical address, LinkedIn/GitHub/"
            "portfolio URLs, website) even if they appear physically close to the languages "
            "section in the raw text — those belong to contact info, not languages. Only "
            "include entries that name an actual language (e.g. 'Spanish', 'English', "
            "'Mandarin'). Empty list if none found."
        )
    )

MODEL_NAME = "qwen2.5:7b-instruct"


class CV_Resumme(BaseModel):
    summary: Optional[str] = Field(
        None,
        description=(
            """A short description or high level statement of who the candidate is professionally as a whole.
            Often found under headings such as 'About me', 'Summary', 'Profile', 'Professional Profile', 
            ,'Professional Summary', 'Objective', 'Perfil', but very frequently has NO heading at all.
            
            Because PDF text extraction can scramble section order, this paragraph may end up
            anywhere in the raw text, not necessarily near the candidate's name or at the top 
            so do NOT rely on its position.
            """
        ),
    )
    work_experience: list[str] = Field(
        default_factory=list,
        description=(
        "Professional roles, jobs and internships held by the candidate. "
        "The main objective is to preserve what the candidate actually did: "
        "responsibilities, tasks, achievements, technical methods, tools used, "
        "problems solved and results obtained. One list entry per distinct role. "
        "Begin each entry with the job title when available, followed by a "
        "detailed free-text description of the candidate's work. "
        "Do NOT include company names, cities, locations or employment dates. "
        "Do NOT split the responsibilities of one role into multiple entries. "
        "Do NOT move normal employment responsibilities into projects. "
        "Do NOT repeat experiences, even if different dates appear. "
        "Empty list if none found."
        ),
    )
    projects: list[str] = Field(
        default_factory=list,
        description=(
            "Specific projects the candidate describes in detail (what was built, tools used, "
            "outcome/impact), as opposed to a general job summary. Covers headings like "
            "'Key Projects', 'Projects', 'Selected Projects', 'Case Studies', 'Portfolio', "
            "'Proyectos Destacados'. Also include personal/side/academic projects if described "
            "with this level of detail. One list entry per distinct project, each as free text. "
            "DO NOT repeat projects even if dates are different."
            "Empty list if none found."
        ),
    )
    technical_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Technical/hard skills: programming languages, frameworks, libraries, tools, "
            "software, platforms, databases, cloud services, methodologies (e.g. 'Python', "
            "'AWS', 'Docker', 'Scrum'). Covers headings like 'Technical Skills', 'Skills', "
            "'Tech Stack', 'Tools', 'Programming Languages', 'Habilidades Técnicas', "
            "'Conocimientos'. Do NOT include soft skills (e.g. 'leadership', 'communication'). "
            "One list entry per skill/tool. Empty list if none found."
        ),
    )
    certifications: list[str] = Field(
        default_factory=list,
        description=(
            "Professional certifications, licenses, or certified courses the candidate holds "
            "(e.g. 'AWS Certified Solutions Architect', 'PMP'). Covers headings like "
            "'Certifications', 'Licenses', 'Certificates', 'Certificaciones'. Include issuer "
            "and/or date if available in the same entry. One list entry per certification. "
            "Empty list if none found."
        ),
    )
    languages: list[str] = Field(
        default_factory=list,
        description=(
            "Spoken/written languages the candidate knows. One list entry per language, NOT a "
            "single combined string. If a proficiency level is stated, format the entry as "
            "'Language: Level' (e.g. 'English: Native', 'Spanish: Professional working "
            "proficiency'). If no level is stated for a language, just include the language "
            "name on its own (e.g. 'French'). Never merge multiple languages into one entry "
            "like 'Spanish, English, Italian' — always split them. Do NOT include contact "
            "details here (email address, phone number, physical address, LinkedIn/GitHub/"
            "portfolio URLs, website) even if they appear physically close to the languages "
            "section in the raw text — those belong to contact info, not languages. Only "
            "include entries that name an actual language (e.g. 'Spanish', 'English', "
            "'Mandarin'). Empty list if none found."
        )
    )

# SYSTEM_PROMPT = """You are a precise CV/resume information extractor.

# You will be given the raw text of a CV extracted from a PDF. Because PDF text \
# extraction reads column-by-column or block-by-block, the text is often OUT OF ORDER: \
# a section's heading (e.g. "ABOUT ME", "KEY PROJECTS") may appear far from its own \
# content, sometimes even after it. Do not rely on linear/reading order — use semantic \
# understanding to figure out which paragraph belongs to which section.

# Extract ONLY the fields defined in the provided JSON schema. Rules:

# - Do not invent, infer, or guess information that is not explicitly present in the text.
# - Preserve the original wording of the CV as much as possible instead of paraphrasing.
# - Reassemble scrambled sections: match orphaned paragraphs to the section heading they \
# describe, even if they are not adjacent in the raw text.
# - Rely on each field's description in the schema to know what belongs in it — it \
# already lists the possible section headings/synonyms for that field.
# - "work_experience" and "projects" are separate fields: work_experience is the \
# high-level job history (company, role, dates, brief summary), while projects holds \
# the more detailed, project-specific descriptions, even if both appear under a single \
# heading in the CV.
# - If a field/section is not present in the CV, leave it empty (null or empty list, \
# as appropriate for that field) instead of making something up.
# - The CV may be in Spanish, English, or a mix of both; extract regardless of language.
# - Return ONLY the JSON object matching the schema, nothing else.
# """

# SYSTEM_PROMPT = """You are a precise CV/resume information extractor.
 
# You will be given the raw text of a CV extracted from a PDF. Because PDF text \
# extraction reads column-by-column or block-by-block, the text is often OUT OF ORDER: \
# a section's heading (e.g. "ABOUT ME", "KEY PROJECTS") may appear far from its own \
# content, sometimes even after it. Do not rely on linear/reading order — use semantic \
# understanding to figure out which paragraph belongs to which section.
 
# Extract ONLY the fields defined in the provided JSON schema. Rules:
 
# - Do not invent, infer, or guess information that is not explicitly present in the text.
# - Preserve the original wording of the CV as much as possible instead of paraphrasing.
# - Reassemble scrambled sections: match orphaned paragraphs to the section heading they \
# describe, even if they are not adjacent in the raw text.
# - Rely on each field's description in the schema to know what belongs in it — it \
# already lists the possible section headings/synonyms for that field.
# - "work_experience" and "projects" are separate fields: work_experience is the \
# high-level job history (company, role, dates, brief summary), while projects holds \
# the more detailed, project-specific descriptions, even if both appear under a single \
# heading in the CV.
# - Every entry in "work_experience" and "projects" must be unique. If two entries \
# describe the same role/company or the same project with the same or near-identical \
# description text, and only the dates differ, treat them as ONE entry, not two — pick \
# the widest/combined date range and keep a single entry. Do not output duplicate or \
# near-duplicate entries just because the CV lists overlapping date ranges for the same work.
# - If a field/section is not present in the CV, leave it empty (null or empty list, \
# as appropriate for that field) instead of making something up.
# - The CV may be in Spanish, English, or a mix of both; extract regardless of language.
# - Return ONLY the JSON object matching the schema, nothing else.
# """

SYSTEM_PROMPT = """You are a precise CV/resume information extractor.
 
You will be given the raw text of a CV extracted from a PDF. Because PDF text \
extraction reads column-by-column or block-by-block, the text is often OUT OF ORDER: \
a section's heading (e.g. "ABOUT ME", "KEY PROJECTS") may appear far from its own \
content, sometimes even after it. Do not rely on linear/reading order — use semantic \
understanding to figure out which paragraph belongs to which section.
 
Extract ONLY the fields defined in the provided JSON schema. Rules:
 
- Do not invent, infer, or guess information that is not explicitly present in the text.
- Preserve the original wording of the CV as much as possible instead of paraphrasing.
- Reassemble scrambled sections: match orphaned paragraphs to the section heading they \
describe, even if they are not adjacent in the raw text.
- Rely on each field's description in the schema to know what belongs in it — it \
already lists the possible section headings/synonyms for that field.
- "work_experience" and "projects" are separate fields: work_experience is the \
high-level job history (company, role, dates, brief summary), while projects holds \
the more detailed, project-specific descriptions, even if both appear under a single \
heading in the CV.

- Do NOT include volunteer work, volunteering, or any variant of it (e.g. "Volunteer \
Positions", "Voluntariado") in "work_experience". This schema does not have a field for \
it, so leave it out entirely rather than placing it in work_experience or projects.

- Every entry in "work_experience" and "projects" must be unique. If two entries \
describe the same role/company or the same project with the same or near-identical \
description text, and only the dates differ, treat them as ONE entry, not two — pick \
the widest/combined date range and keep a single entry. Do not output duplicate or \
near-duplicate entries just because the CV lists overlapping date ranges for the same work.

- technical_skills must contain ONLY technical/hard skills (programming languages, \
frameworks, tools, software, platforms, methodologies). Do NOT include soft skills \
(e.g. "leadership", "teamwork", "communication", "problem-solving") even if they appear \
in the same list or section as technical skills in the CV.

- If a field/section is not present in the CV, leave it empty (null or empty list, \
as appropriate for that field) instead of making something up.
- Personal contact details (email address, phone number, physical address, LinkedIn/\
GitHub/portfolio URLs, website) never belong in any of the schema's fields, even if they \
appear physically close to a relevant section in the raw text due to PDF layout. This \
schema does not capture contact info at all — ignore it when extracting.
- The CV may be in Spanish, English, or a mix of both; extract regardless of language.
- Return ONLY the JSON object matching the schema, nothing else.
"""

SYSTEM_PROMPT_MK_2 = """You are a precise CV/resume information extractor.

You will be given a CV extracted as Markdown. The Markdown preserves most of the
original document structure, including headings, subheadings, lists, tables, and
reading order.

Extract ONLY the fields defined in the provided JSON schema.

Rules:

- Use the Markdown structure to identify sections. Section headings, subheadings,
  bullet lists, and tables provide the primary cues for assigning information to
  the correct schema field.
- Preserve the original wording of the CV whenever possible instead of paraphrasing.
- Do not invent, infer, or guess information that is not explicitly present.
- Follow each field's description in the schema to determine what belongs in it.

- summary: The complete paragraph(s) immediately following headings such as
"About Me", "Summary", "Professional Summary", "Profile",
"Professional Profile", "Objective", "Career Summary",
"Resumen Profesional", or similar. This section describes the candidate at a high level and usually appears
before Work Experience. Extract it verbatim until the next section heading.
Do not generate or infer a summary from other sections.

- "work_experience" and "projects" are separate fields:
  - work_experience contains the candidate's employment history (company, role,
    dates, short description).
  - projects contains detailed project descriptions, personal projects,
    academic projects, or project-focused work, even if they appear inside a
    work experience section.

- Do NOT include volunteer work (or equivalent headings such as "Volunteer",
  "Volunteering", "Volunteer Positions", "Voluntariado") in either
  "work_experience" or "projects". Ignore those sections completely.

- Every entry in "work_experience" and "projects" must be unique.
  If multiple entries refer to the same role/company or the same project with
  nearly identical descriptions but different dates, merge them into a single
  entry using the widest applicable date range. 

- technical_skills must contain ONLY technical/hard skills such as:
  programming languages, frameworks, libraries, databases, cloud platforms,
  software, tools, technologies, methodologies, operating systems and
  development environments.

- Never include soft skills (leadership, teamwork, communication,
  problem-solving, adaptability, etc.) in technical_skills, even if they appear
  in the same section.

- Ignore all personal contact information. Do NOT extract:
  - email addresses
  - phone numbers
  - postal addresses
  - LinkedIn URLs
  - GitHub URLs
  - portfolio/personal websites
  - social media profiles

- If a field is not present, leave it empty (null or an empty list, depending on
  the schema).

- The CV may be written in English, Spanish, or a mixture of both.

Return ONLY the JSON object that matches the provided schema.
"""

SYSTEM_PROMPT_MK = """You are a precise CV/resume information extractor.

You will be given a CV extracted as Markdown. The Markdown preserves parts of
the original document structure, but content from multi-column CVs may appear
out of visual order. Headings, companies, dates, job titles and descriptions
may therefore be separated.

Extract ONLY the fields defined in the provided JSON schema.

General rules:

- Preserve the original wording of the CV whenever possible.
- Do not invent, infer or guess information that is not explicitly present.
- Use headings, item order, dates and semantic relationships to reconstruct
  sections when the Markdown reading order is imperfect.
- Follow each field description in the JSON schema.
- Ignore HTML comments such as <!-- image -->.
- Ignore warning messages, logs or processing metadata appearing before or
  after the CV.
- Ignore anonymization placeholders used as contact information, such as
  <EMAIL_ADDRESS>, <PHONE_NUMBER> and <URL>.

SUMMARY

- Extract only the complete paragraph or paragraphs immediately following
  headings such as "About Me", "Summary", "Professional Summary", "Profile",
  "Professional Profile", "Objective", "Career Summary",
  "Resumen Profesional", or equivalent.
- Extract it verbatim until the next section heading.
- Do not generate a summary from other sections.

WORK EXPERIENCE

- work_experience contains employment history.
- Each entry should include, when available:
  company, job title, location, start date, end date, responsibilities and
  achievements.
- Responsibilities and achievements written below a job title belong to that
  work experience entry.
- Do not move employment responsibilities into projects merely because they
  describe building a model, application, pipeline, experiment, analysis or
  technical solution.
- Bullet points beginning with verbs such as "developed", "created",
  "engineered", "conducted", "applied", "managed", "implemented" or "designed"
  should remain under the associated employment entry when they describe work
  performed in that role.
- Preserve all relevant responsibility and achievement bullet points.
- Do not reduce a work experience entry to only company, title and dates.

MULTI-COLUMN RECONSTRUCTION

- The CV may list companies and dates first, followed later by job titles and
  descriptions because of multi-column extraction.
- When this occurs, reconstruct the entries using their sequence and context.
- Pair companies with job titles and descriptions in corresponding order when
  the document provides no stronger conflicting evidence.
- For example, four companies followed by four job titles should normally be
  paired first-to-first, second-to-second, third-to-third and fourth-to-fourth.
- A job description belongs to the closest associated job title, not to the
  projects field.

PROJECTS

- projects contains only:
  - entries under an explicit heading such as "Projects", "Personal Projects",
    "Academic Projects", "Selected Projects" or equivalent; or
  - clearly named standalone projects explicitly presented as projects.
- Do not classify normal employment responsibilities as projects.
- Do not create a project from individual work-experience bullet points.
- If no explicit or clearly identifiable project is present, return an empty
  projects list.

DEDUPLICATION

- Every work experience and project entry must be unique.
- Merge entries only when they clearly refer to the same role at the same
  company or to the same named project.
- Do not merge different roles merely because they share similar technical
  responsibilities.

VOLUNTEERING

- Do not include volunteer work under work_experience or projects.
- Ignore sections such as "Volunteer", "Volunteering", "Volunteer Positions"
  and "Voluntariado".

TECHNICAL SKILLS

- technical_skills must contain only technical or hard skills explicitly
  present in the CV.
- Valid examples include programming languages, frameworks, libraries,
  databases, cloud platforms, software, tools, technologies, methodologies,
  operating systems and development environments.
- Never include soft skills.
- Do not derive technical skills from work descriptions unless the technology
  is explicitly named.
- Do not include broad professional areas such as "Data Science" or
  "Artificial Intelligence" unless they are explicitly listed as skills.

CONTACT INFORMATION

Ignore and do not extract:

- email addresses
- phone numbers
- postal addresses
- LinkedIn URLs
- GitHub URLs
- portfolio or personal websites
- social media profiles

MISSING DATA

- If a field is absent, return null or an empty list according to the schema.
- Do not create placeholder entries.

The CV may be written in English, Spanish or a mixture of both.

Return ONLY the JSON object matching the provided schema.
"""
SYSTEM_PROMPT_MK = """
You are a precise CV information extractor.

The input is Markdown extracted from a CV. Because the original CV may have
multiple columns, companies, dates, job titles and descriptions may appear
separated or out of visual order.

Extract only the fields defined in the provided JSON schema.

GENERAL RULES

- Extract only information explicitly present in the CV.
- Never invent, infer or restore missing information.
- Preserve placeholders exactly as written, including:
  <PERSON>, <ORGANIZATION>, <LOCATION>, <DATE_TIME>, <URL>,
  <PHONE_NUMBER> and <EMAIL_ADDRESS>.
- Ignore logs, warnings and HTML comments such as <!-- image -->.
- Preserve the original wording whenever possible.
- Return only the JSON object matching the schema.

SUMMARY

- Extract only text appearing directly below headings such as:
  "About Me", "Summary", "Profile", "Professional Summary",
  "Professional Profile" or equivalent.
- Stop at the next section heading.
- Do not generate a summary from other CV sections.

WORK EXPERIENCE

- Each work experience entry must contain, when available:
  company, job title, location, dates and responsibilities.
- Responsibilities and achievements written below a job title always belong
  to that work experience entry.
- Never move employment bullet points to the projects field.
- Statements describing models, pipelines, analyses, experiments, software,
  applications or technical solutions remain work responsibilities when they
  appear under a job title.
- Preserve every responsibility as a separate list item.

MULTI-COLUMN RECONSTRUCTION

- Companies and dates may appear first, followed later by job titles and their
  descriptions.
- When no stronger evidence exists, pair companies and job titles by order:
  first company with first job title, second company with second job title,
  and so on.
- Bullet points belong to the closest job title immediately preceding them.
- Do not associate education descriptions with work experience.

PROJECTS

- Extract only projects appearing under an explicit heading such as:
  "Projects", "Personal Projects", "Academic Projects",
  "Selected Projects" or equivalent.
- A project may also be extracted when it is explicitly identified as a
  standalone named project.
- Never create projects from work-experience responsibilities.
- If there is no explicit project section or named project, return an empty
  list.

TECHNICAL SKILLS

- Extract only explicitly mentioned technical skills.
- Include programming languages, frameworks, libraries, databases, cloud
  platforms, software, tools, technologies and technical methodologies.
- Do not include soft skills.
- Do not derive a skill merely because it could be inferred from a job title.

OTHER RULES

- Ignore volunteer experience.
- Ignore contact information and personal URLs.
- If information is missing, use null or an empty list according to the schema.
- The CV may be written in English, Spanish or both.

Return only valid JSON matching the provided schema.
"""

SYSTEM_PROMPT_MK_3 = """
You are a precise CV information extractor.

The input is Markdown extracted from a CV. Because the original CV may have
multiple columns, companies, dates, job titles and descriptions may appear
separated or out of visual order.

Extract only the fields defined in the provided JSON schema.

GENERAL RULES

- Extract only information explicitly present in the CV.
- Never invent, infer or restore missing information.
- Preserve placeholders exactly as written, including:
  <PERSON>, <ORGANIZATION>, <LOCATION>, <DATE_TIME>, <URL>,
  <PHONE_NUMBER> and <EMAIL_ADDRESS>.
- Ignore logs, warnings and HTML comments such as <!-- image -->.
- Preserve the original wording whenever possible.
- Return only the JSON object matching the schema.

SUMMARY

-The complete paragraph(s) immediately following headings such as
"About Me", "Summary", "Professional Summary", "Profile",
"Professional Profile", "Objective", "Career Summary",
"Resumen Profesional", or similar. This section describes the candidate at a high level and usually appears
before Work Experience. Extract it verbatim until the next section heading.
Do not generate or infer a summary from other sections.

WORK EXPERIENCE

- Each work experience entry must contain, when available:
  company, job title, location, dates and responsibilities.
- Responsibilities and achievements written below a job title always belong
  to that work experience entry.
- Never move employment bullet points to the projects field.
- Statements describing models, pipelines, analyses, experiments, software,
  applications or technical solutions remain work responsibilities when they
  appear under a job title.
- Preserve every responsibility as a separate list item.

MULTI-COLUMN RECONSTRUCTION

- Companies and dates may appear first, followed later by job titles and their
  descriptions.
- When no stronger evidence exists, pair companies and job titles by order:
  first company with first job title, second company with second job title,
  and so on.
- Bullet points belong to the closest job title immediately preceding them.
- Do not associate education descriptions with work experience.

PROJECTS

- Extract only projects appearing under an explicit heading such as:
  "Projects", "Personal Projects", "Academic Projects",
  "Selected Projects" or equivalent.
- A project may also be extracted when it is explicitly identified as a
  standalone named project.
- Never create projects from work-experience responsibilities.
- If there is no explicit project section or named project, return an empty
  list.

TECHNICAL SKILLS

- Extract only explicitly mentioned technical skills.
- Include programming languages, frameworks, libraries, databases, cloud
  platforms, software, tools, technologies and technical methodologies.
- Do not include soft skills.
- Do not derive a skill merely because it could be inferred from a job title.

CERTIFICATIONS

- Extract only certifications, professional certificates and licenses that are
  explicitly present in the CV.
- Include the certification name and, when explicitly available:
  issuing organization, issue date, expiration date and credential ID.
- Preserve the original wording.
- Do not include university degrees, academic qualifications, publications,
  awards, conferences or normal training activities as certifications.
- Do not infer that a course is a certification unless the CV explicitly
  presents it as one.
- If no certification section or explicit certification is present, return an
  empty list.

LANGUAGES

- Extract only human languages explicitly mentioned in the CV.
- Preserve the proficiency level when explicitly provided, for example:
  Native, Fluent, Professional, Intermediate, B2 or C1.
- Do not infer proficiency from the language used to write the CV.
- Do not infer languages from nationality, location, education or employment.
- Do not include programming languages such as Python, Java, SQL or JavaScript.
- Preserve the original wording.
- If only the language name is provided, extract only the language name.
- If no human languages are explicitly present, return an empty list.

OTHER RULES

- Ignore volunteer experience.
- Ignore contact information and personal URLs.
- If information is missing, use null or an empty list according to the schema.
- The CV may be written in English, Spanish or both.

Return only valid JSON matching the provided schema.
"""

# Few-shot example: a real CV with scrambled PDF extraction order, showing how
# "ABOUT ME" content appears before its own heading, and how WORK EXPERIENCE (brief,
# job-level) and KEY PROJECTS (detailed, project-level) map to separate fields.
EXAMPLE_INPUT = """DAVID DE LAS HERAS
PERSONAL SKILLS
Degree in Mathematics
Engineering
Universidad Complutense de Madrid
2016 - 2020
EDUCATION
Business impact through end-to-end AI and ML solutions, from design to deployment.
Passionate about solving business challenges through data and AI, with more than 6 \
years of experience at Accenture delivering advanced analytics and machine learning \
solutions across industries such as Telco, Financial Services, Resources, and \
Products. Specialized in the design and implementation of scalable forecasting \
systems, MLOps architectures, and Generative AI use cases.
ABOUT ME
WORK EXPERIENCE
2019 - On going
Data & AI Data Scientist
Accenture - Strategy & Consulting
Application of machine learning and analytics techniques to address complex \
business problems, working closely with stakeholders to align data solutions with \
strategic goals.
Data & AI Intern
2017 - 2018
Accenture - Strategy & Consulting
ZAI Program
KEY PROJECTS
Built attribution models to quantify the impact of corporate events and marketing \
initiatives on revenue and sales performance.
Attribution Modeling for Business Impact (2019-2020)
Designed and deployed predictive models such as churn and upsell propensity to \
generate optimized customer segments for targeted marketing actions.
Predictive Targeting for Marketing Campaigns (2020-2022)"""

EXAMPLE_OUTPUT = CV(
    summary=(
        "Passionate about solving business challenges through data and AI, with "
        "more than 6 years of experience at Accenture delivering advanced analytics "
        "and machine learning solutions across industries such as Telco, Financial "
        "Services, Resources, and Products. Specialized in the design and "
        "implementation of scalable forecasting systems, MLOps architectures, and "
        "Generative AI use cases."
    ),
    work_experience=[
        """
        2019 – On going
        Accenture - Strategy & Consulting
        Data & AI Data Scientist: Application of machine learning and analytics techniques to
        address complex business problems, working closely with
        stakeholders to align data solutions with strategic goals
        """,
        """
        2024 - On going
        Universidad Alfonso X el Sabio Univerity professor Professor in the Bachelor's degree in Artificial Intelligence and Computer Science.
        """,
        """
        2017 - 2018 Accenture - Strategy & Consulting
        Data & AI Intern
        Internship focused on supporting data analysis and machine
        learning initiatives, contributing to the development of early-
        stage models.
        """
    ],
    projects=[
        """
        Generative AI & NLP Training (2025):
        Delivered hands-on training sessions for a Bank, covering topics
        from traditional NLP to Retrieval-Augmented Generation (RAG)
        and AI Agents, with a focus on real-world applications
        """,
        """
        Generative AI for Data Quality Automation (2024-2025):
        Developed a PoC to automate the generation of data quality
        rules, ingestion code, documentation, and AWS infrastructure
        based on table metadata. Leveraged GenAI to streamline
        repetitive tasks and accelerate onboarding of new data assets.
        """,

        """Attribution Modeling for Business Impact (2019-2020): Built attribution 
        models to quantify the impact of corporate events and marketing initiatives
        on revenue and sales performance.""",

        """Predictive Targeting for Marketing Campaigns (2020-2022): Designed and "
        "deployed predictive models such as churn and upsell propensity to generate "
        "optimized customer segments for targeted marketing actions.""",
    ],
).model_dump_json()

SYSTEM_PROMPT_MK_FULL = """
You are a precise CV/resume information extractor.

You will receive a CV extracted as Markdown. The Markdown may preserve headings,
subheadings, bullet lists and tables, but the original document may have multiple
columns. Therefore, companies, dates, job titles and descriptions may appear
separated or out of visual order.

Extract only the fields defined in the provided JSON schema.

Return only a valid JSON object matching the schema.

GENERAL RULES

- Extract only information explicitly present in the CV.
- Do not invent, guess, assume or reconstruct missing information.
- Preserve the original wording and text whenever possible.
- The CV may be written in English, Spanish or a mixture of both

- Preserve anonymization placeholders exactly as written, including:

  <PERSON>
  <ORGANIZATION>
  <LOCATION>
  <DATE_TIME>
  <URL>
  <PHONE_NUMBER>
  <EMAIL_ADDRESS>

- Never guess the original information hidden behind an anonymization
  placeholder.
- Ignore duplicated content caused by PDF extraction.
- Do not return the same information more than once.
- If a field is genuinely absent:
  - return null for summary;
  - return an empty list for every list field.

EXTRACTION ORDER

Extract and verify the fields in this order:

1. summary
2. work_experience
3. projects
4. technical_skills
5. certifications
6. languages

Do not skip an earlier field merely because later sections are easier to
identify.

SUMMARY

- Extract a short high-level professional description of the candidate.
- The summary should normally contain between one and four sentences.
- It describes who the candidate is professionally as a whole, such as:
  - their professional role;
  - their general expertise;
  - their overall experience;
  - their professional interests;
  - their career objective;
  - an aspirational statement such as "passionate about..." or
    "committed to...".

- Look for headings such as:

  "About Me"
  "Summary"
  "Professional Summary"
  "Profile"
  "Professional Profile"
  "Objective"
  "Career Summary"
  "Sobre mí"
  "Perfil"
  "Resumen"
  "Resumen Profesional"

- If one of these headings exists and a professional paragraph follows it,
  summary must not be null or empty.
- Extract the paragraph or paragraphs after the heading until the next CV
  section begins.
- Ignore HTML comments or blank lines between the heading and the summary text.
- The summary may also appear without an explicit heading.
- Because multi-column PDF extraction may scramble the reading order, the
  summary may appear anywhere in the Markdown.
- Identify an unheaded summary by its content, not only by its position.
- An unheaded summary is a generic, high-level description of the candidate.
- It should not describe:
  - one specific company;
  - one specific job;
  - one specific project;
  - one specific degree;
  - one date range.

- Do not generate a summary by combining information from work experience,
  education, skills or projects.
- Extract a summary only when a real summary paragraph is explicitly present.

WORK EXPERIENCE

- work_experience must contain the candidate's professional roles, jobs and
  internships.
- The main objective is to extract what the candidate actually did in each
  professional role.

For each role, prioritize:

1. responsibilities;
2. tasks performed;
3. achievements;
4. technical activities;
5. methodologies used;
6. tools and technologies used;
7. problems solved;
8. results or impact obtained.

- Company names, cities, locations and employment dates are not required in the
  output.
- Do not include company names in work_experience entries.
- Do not include cities or locations in work_experience entries.
- Do not include employment dates in work_experience entries.
- Begin each entry with the job title when it is explicitly available.
- After the job title, include the candidate's responsibilities, achievements
  and relevant technical work.
- Create one list entry per distinct professional role.
- Keep all responsibilities belonging to the same role inside the same string.
- Do not create one list entry per responsibility.
- Do not reduce an experience to only the job title when responsibilities are
  available.
- Preserve all relevant responsibility and achievement bullet points.
- Do not replace detailed responsibilities with a vague summary.
- Preserve the original meaning and wording whenever possible.

Statements beginning with verbs such as the following usually describe work
responsibilities when they appear under a professional role:

- developed;
- created;
- designed;
- implemented;
- engineered;
- conducted;
- applied;
- built;
- analyzed;
- managed;
- led;
- evaluated;
- optimized;
- automated;
- deployed;
- maintained;
- researched;
- presented.

- Activities involving models, pipelines, software, applications, analysis,
  experiments, lectures, reporting or technical solutions remain part of
  work_experience when they were performed as part of a professional role.
- Never move normal employment responsibilities into projects.
- Do not include volunteer experience.
- Ignore sections such as:

  "Volunteer"
  "Volunteering"
  "Volunteer Experience"
  "Volunteer Positions"
  "Voluntariado"

- Do not repeat the same professional role.
- If duplicated entries clearly describe the same role, merge their
  responsibilities into one entry.
- Do not merge different roles merely because they contain similar tasks.

MULTI-COLUMN WORK EXPERIENCE RECONSTRUCTION

- In multi-column CVs, companies and dates may appear first, while job titles
  and descriptions may appear later.
- Job titles and responsibilities may therefore be separated from their
  contextual company headings.
- Reconstruct the relationship using:
  - section headings;
  - document order;
  - semantic meaning;
  - the number of entries;
  - proximity between job titles and descriptions.

- When there is no stronger conflicting evidence, pair corresponding elements
  by order:
  - first company block with first job title;
  - second company block with second job title;
  - third company block with third job title;
  - and so on.

- Bullet points belong to the closest relevant job title preceding them until:
  - another job title begins;
  - another section begins;
  - the content clearly changes to education, projects or another category.

- Even when the company or date cannot be confidently identified, preserve the
  job title and all associated responsibilities.
- The most important requirement is not to lose what the candidate did.

PROJECTS

- projects must contain only clearly identifiable projects.
- Extract projects appearing under headings such as:

  "Projects"
  "Key Projects"
  "Selected Projects"
  "Personal Projects"
  "Academic Projects"
  "Case Studies"
  "Portfolio"
  "Proyectos"
  "Proyectos Destacados"
  "Proyectos Personales"
  "Proyectos Académicos"

- A project may also be extracted without a projects heading when it is
  explicitly presented as a named standalone project.
- Include personal, academic and side projects when they are explicitly
  described as projects.
- Each project entry should contain, when available:
  - project name;
  - what was built or developed;
  - tools or technologies used;
  - objective;
  - result or impact.

- Create one list entry per distinct project.
- Keep all information belonging to the same project inside the same string.
- Do not create projects from normal work-experience responsibilities.
- Do not classify a task as a project merely because it mentions:
  - developing a model;
  - creating a pipeline;
  - building software;
  - performing an analysis;
  - conducting an experiment;
  - implementing a technical solution.

- If these activities were performed under a job title, they belong to
  work_experience.
- If no explicit or clearly identifiable project exists, return an empty list.
- Do not repeat the same project.
- Merge duplicated descriptions only when they clearly refer to the same
  project.

TECHNICAL SKILLS

- technical_skills must contain only technical or hard skills explicitly
  mentioned in the CV.
- Include items such as:
  - programming languages;
  - frameworks;
  - libraries;
  - databases;
  - cloud platforms;
  - software;
  - tools;
  - technologies;
  - machine learning techniques;
  - data methodologies;
  - development methodologies;
  - operating systems;
  - development environments.

Examples of valid technical skills include:

- Python
- SQL
- Java
- JavaScript
- PyTorch
- TensorFlow
- Pandas
- Docker
- Kubernetes
- Azure
- AWS
- PostgreSQL
- Machine Learning
- Natural Language Processing
- Clustering
- Time Series Analysis
- A/B Testing
- Statistical Analysis
- Data Pipelines
- Scrum

- A technical skill may be extracted from a skills section or from another CV
  section when the technology or methodology is explicitly named.
- Do not infer technologies that are not explicitly mentioned.
- Do not include job titles as technical skills.
- Do not include broad personal qualities.
- Never include soft skills such as:
  - leadership;
  - teamwork;
  - communication;
  - adaptability;
  - creativity;
  - problem-solving;
  - critical thinking;
  - organization;
  - time management.

- Do not include human languages such as English, Spanish or German in
  technical_skills.
- Create one list entry per skill.
- Do not combine several skills into one string when they can be separated.
- Remove duplicates while preserving the original terminology.

CERTIFICATIONS

- certifications must contain only explicitly mentioned professional
  certifications, licenses, certificates or certified courses.
- Look for headings such as:

  "Certifications"
  "Certificates"
  "Licenses"
  "Courses and Certifications"
  "Professional Certifications"
  "Certificaciones"
  "Certificados"
  "Licencias"

- Each certification must be one separate list entry.
- Include, when explicitly available:
  - certification name;
  - issuing organization;
  - issue date;
  - expiration date;
  - credential ID.

- Keep these details inside the same string.
- Preserve the original wording.
- Do not include:
  - university degrees;
  - master's degrees;
  - bachelor's degrees;
  - academic qualifications;
  - publications;
  - awards;
  - conferences;
  - normal work training;
  - general courses that are not presented as certifications.

- Do not infer that a course is certified unless the CV explicitly presents it
  as a certification or certificate.
- If no certification is explicitly present, return an empty list.
- Do not repeat the same certification.

LANGUAGES

- languages must contain only human languages explicitly mentioned in the CV.
- Look for headings such as:

  "Languages"
  "Language Skills"
  "Idiomas"
  "Lenguas"

- Create one list entry per language.
- Never combine multiple languages into a single string.
- If a proficiency level is explicitly provided, use this format:

  "Language: Level"

Examples:

  "English: Native"
  "Spanish: Professional working proficiency"
  "German: B2"
  "French: Intermediate"

- If no proficiency level is provided, return only the language name.

Examples:

  "English"
  "Spanish"
  "German"

- Do not infer proficiency from:
  - the language used to write the CV;
  - nationality;
  - location;
  - education;
  - employment history.

- Do not include programming languages such as:
  - Python;
  - Java;
  - SQL;
  - JavaScript;
  - C++;
  - R.

- Do not include contact information even when it appears close to the language
  section.
- Do not include:
  - email addresses;
  - phone numbers;
  - physical addresses;
  - LinkedIn URLs;
  - GitHub URLs;
  - portfolio URLs;
  - personal websites;
  - social media profiles.

- Remove duplicated languages.
- If no human language is explicitly present, return an empty list.

CONTACT INFORMATION

Ignore all personal contact information.

Do not extract:

- email addresses;
- phone numbers;
- postal addresses;
- postal codes;
- LinkedIn URLs;
- GitHub URLs;
- portfolio URLs;
- personal websites;
- social media profiles.

EXPECTED OUTPUT EXAMPLE

The following example illustrates only the required JSON structure and level of
detail.

Do not copy any information from this example unless it is explicitly present
in the input CV.

{
  "summary": "Data professional with experience developing analytical and machine learning solutions. Passionate about applying data to complex business problems.",

  "work_experience": [
    "Data Scientist. Developed machine learning models to solve business problems. Created customer segmentation models using clustering techniques. Designed and evaluated A/B tests. Applied statistical analysis and presented findings to business stakeholders.",
    "Visiting Lecturer in Data Science. Prepared master's-level lectures on data science applied to financial risk management. Led practical sessions for graduate students.",
    "Big Data Engineer. Developed data ingestion pipelines and integrated information from heterogeneous sources. Built software solutions for processing large datasets.",
    "Engineering Intern. Supported technical design and analysis activities."
  ],

  "projects": [
    "Customer Churn Prediction. Developed a classification model to identify customers at risk of leaving. Performed data preprocessing, feature engineering and model evaluation using Python and scikit-learn."
  ],

  "technical_skills": [
    "Python",
    "SQL",
    "Machine Learning",
    "Clustering",
    "Time Series Analysis",
    "A/B Testing",
    "Statistical Analysis",
    "Data Pipelines"
  ],

  "certifications": [
    "AWS Certified Cloud Practitioner"
  ],

  "languages": [
    "Spanish: Native",
    "English: Professional working proficiency",
    "German"
  ]
}

OUTPUT FORMAT RULES

- Return exactly these fields:
  - summary
  - work_experience
  - projects
  - technical_skills
  - certifications
  - languages

- summary must be a string or null.
- work_experience must be a list of strings.
- projects must be a list of strings.
- technical_skills must be a list of strings.
- certifications must be a list of strings.
- languages must be a list of strings.

- Each work_experience string represents one distinct professional role.
- Each projects string represents one distinct project.
- Each technical_skills string represents one distinct technical skill.
- Each certifications string represents one distinct certification.
- Each languages string represents one distinct human language.
- Never return objects inside these lists.
- Never add fields that are not defined in the schema.
- Never include Markdown formatting around the JSON.
- Never include explanations before or after the JSON.

FINAL VALIDATION

Before returning the JSON, verify all of the following:

1. If an explicit "About Me", "Summary", "Profile" or equivalent paragraph is
   present, summary contains it.

2. summary was not generated from unrelated CV sections.

3. work_experience prioritizes what the candidate actually did.

4. Every available work responsibility remains inside the corresponding
   work_experience entry.

5. Company names, locations and dates were removed from work_experience.

6. Normal employment responsibilities were not moved into projects.

7. projects contains only explicit or clearly identified standalone projects.

8. technical_skills contains only technical or hard skills.

9. certifications does not contain degrees, publications or awards.

10. languages contains one human language per list entry.

11. Contact information was not extracted.

12. Every field has the type required by the JSON schema.

Return only the final valid JSON object.
"""


PROMPT_MK_FULL_2 = """
You are a precise CV/resume information extractor.

You will receive a CV extracted as Markdown. The Markdown may preserve headings,
subheadings, bullet lists and tables, but the original document may have multiple
columns. Therefore, companies, dates, job titles and descriptions may appear
separated or out of visual order.

Extract only the fields defined in the provided JSON schema.

Return only a valid JSON object matching the schema.

GENERAL RULES

- Extract only information explicitly present in the CV.
- Do not invent, guess, assume or reconstruct missing information.
- Preserve the original wording and text whenever possible.
- The CV may be written in English, Spanish or a mixture of both
- Never guess the original information hidden behind an anonymization
  placeholder.
- Ignore duplicated content caused by PDF extraction.
- Do not return the same information more than once.
- If a field is genuinely absent:
  - return null for summary;
  - return an empty list for every list field.

EXTRACTION ORDER

Extract and verify the fields in this order:

1. summary
2. work_experience
3. projects
4. technical_skills
5. certifications
6. languages

Do not skip an earlier field merely because later sections are easier to
identify.

SUMMARY

- Extract a short high-level professional description of the candidate.
- The summary should normally contain between one and four sentences.
- It describes who the candidate is professionally as a whole, such as:
  - their professional role;
  - their general expertise;
  - their overall experience;
  - their professional interests;
  - their career objective;
  - an aspirational statement such as "passionate about..." or
    "committed to...".

- Look for headings such as:

  "About Me"
  "Summary"
  "Professional Summary"
  "Profile"
  "Professional Profile"
  "Objective"
  "Career Summary"
  "Sobre mí"
  "Perfil"
  "Resumen"
  "Resumen Profesional"

- If one of these headings exists and a professional paragraph follows it,
  summary must not be null or empty.
- Extract the paragraph or paragraphs after the heading until the next CV
  section begins.
- Ignore HTML comments or blank lines between the heading and the summary text.
- The summary may also appear without an explicit heading.
- Because multi-column PDF extraction may scramble the reading order, the
  summary may appear anywhere in the Markdown.
- Identify an unheaded summary by its content, not only by its position.
- An unheaded summary is a generic, high-level description of the candidate.
- It should not describe:
  - one specific company;
  - one specific job;
  - one specific project;
  - one specific degree;
  - one date range.

- Do not generate a summary by combining information from work experience,
  education, skills or projects.
- Extract a summary only when a real summary paragraph is explicitly present.

"""

def parse_cv(cv_text: str, model: str = MODEL_NAME) -> CV:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_INPUT},
            {"role": "assistant", "content": EXAMPLE_OUTPUT},
            {"role": "user", "content": cv_text},
        ],
        format=CV.model_json_schema(),
        options={"temperature": 0},
    )
    return CV.model_validate_json(response["message"]["content"])

def parse_cv_markdown(cv_text: str, model: str = MODEL_NAME) -> CV:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT_MK_FULL_2},
            {"role": "user", "content": cv_text},
        ],
        format=CV.model_json_schema(),
        options={"temperature": 0},
    )
    return CV.model_validate_json(response["message"]["content"])

from pathlib import Path

def save_txt(lines: str | list[str], path: str | Path) -> None:
    """
    Guarda texto en un fichero.
    
    Args:
        lines: Un string o una lista de strings.
        path: Ruta del fichero de salida.
    """
    ruta = Path(path)

    if isinstance(lines, str):
        lines = [lines]

    with ruta.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n\n")

# def save_object_to_txt(obj, output_path: str | Path) -> None:
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     fields = obj.model_dump()

#     sections = []

#     for field_name, value in fields.items():
#         if value is None:
#             continue

#         section = (
#             f"{field_name.upper()}\n"
#             f"{value}"
#         )

#         sections.append(section)

#     content = "\n\n" + "\n\n"  # placeholder
#     content = "\n\n" + ("-" * 80) + "\n\n"
#     content = content.join(sections)

#     output_path.write_text(
#         content,
#         encoding="utf-8",
#     )


def save_object_to_txt(obj, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    for field_name, value in obj.model_dump().items():
        if value is None:
            continue

        if isinstance(value, list):
            formatted_value = "\n".join(
                f"- {item}" for item in value
            )
        else:
            formatted_value = str(value)

        sections.append(
            f"{field_name.upper()}\n{formatted_value}"
        )

    separator = "\n\n" + "-" * 80 + "\n\n"

    output_path.write_text(
        separator.join(sections),
        encoding="utf-8",
    )

PROMPT_JOB_DESCRIPTION = """
You are an expert in job description analysis.

Your task is to extract all the content related to "Responsibilities and Requirements" from a raw job description.

This category must include:

   - Tasks and duties the candidate will perform.
   - Activities, functions, and areas of responsibility associated with the role.
   - Projects, objectives, or business applications the candidate will work on.
   - Required or preferred professional experience.
   - Required knowledge or technical experience.
   - Requiered technical capacities

Exclude:
- Company description, mission, values, or culture.
- Statements about career development or personal growth.
- Benefits, salary, compensation, insurance, holidays, or working conditions.
- Generic promotional or recruiting language.
- Required proficiency in languages.
- Skills that are only mentioned as isolated competencies without being presented as a requirement or responsibility.

Important rules:
- Extract the relevant information from the original job description; do not invent or infer requirements.
- Preserve the original meaning.
- Each bullet should represent one clear responsibility or requirement.
- Do not merge unrelated responsibilities or requirements into a single bullet.
- Remove redundant introductory text.
- Return only the extracted bullets.
- If there is no relevant information, return an empty list.

Output format:
[
  "bullet 1",
  "bullet 2",
  "bullet 3"
]

Raw job description:
{job_description}
"""

def parse_job_description(job_descript: str, model: str = MODEL_NAME) -> CV:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT_JOB_DESCRIPTION},
            {"role": "user", "content": job_descript},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]