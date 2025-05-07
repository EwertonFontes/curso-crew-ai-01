from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from crewai import Agent, Task, Crew
from markdown import markdown

pesquisador = Agent(
    role="Pesquisador de Mercado",
    goal="Coletar e organizar informações relevantes sobre {sector}",
    backstory="""
    Você é um pesquisador de mercado experiente que analisa tendencias de mercado e coleta
    dados relevantes sobre {sector}. Seu trabalho é garantir que todas as informaçoes
    estejam atualizadas e bem documentadas.
    """,
    allow_delegation=False,
    verbose=True
)

analista = Agent(
    role="Analista de Tendências",
    goal="Analisar dados do setor {sector} e identificar padrões e oportunidades",
    backstory="""
    Você é um analista de mercado que examina os dados coletados para identificar tendencias
    emergentes, oportunidades e ameaças no setor {sector}
    """,
    allow_delegation=False,
    verbose=True
)

redator = Agent(
    role="Um Redator de Relatórios",
    goal="Elaborar um relatório consolidado sobre analise de mercado do setor {sector}",
    backstory="""
    Você é um redator profissional que transforma análises de mercado em um relatório estruturado
    e compreensível para tomadores de decisão
    """,
    allow_delegation=False,
    verbose=True
)

coleta_dados = Task(
    description=(
        "1. Pesquisar e coletar informações atualizadas sobre {sector}."
        "2. Identificar os principais players, tendências e estatísticas do setor."
        "3. Organizar os dados de forma clara para análise"
    ),
    expected_output="Um documento estruturado contendo dados de mercado sobre {sector}",
    agent=pesquisador
)

analise_tendencias = Task(
    description=(
        "1. Examinar os dados coletados pelo Pesquisador de Mercado."
        "2. Identificar padrões, tendências emergentes e oportunidades no setor {sector}"
        "3. Elaborar uma análise detalhada destacando os principais pontos."
    ),
    expected_output="Um relatório com insights e tendências baseados nos dados do setor {sector}",
    agent=analista
)

redacao_relatorio = Task(
    description=(
        "1. Usar a análise de tendências para criar um relatório detalhado sobre {sector}."
        "2. Garantir que o relatório seja bem estruturado e compreensível."
        "3. Apresentar um resumo executivo e recomendações finais"
    ),
    expected_output="Um relatório de análise de mercado em formato Markdown, pronto para leitura e apresentação.",
    agent=redator
)

crew = Crew(
    agents=[pesquisador,analista,redator],
    tasks=[coleta_dados,analise_tendencias,redacao_relatorio],
    verbose=True
)

result = crew.kickoff(inputs={"sector": "Inteligencia Artificial"})

html = markdown(str(result))
with open("artigo.html", "w",  encoding="utf-8") as file:
    file.write(html)