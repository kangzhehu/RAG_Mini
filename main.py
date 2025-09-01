#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/1 14:00
# @Author  : hukangzhe
# @File    : main.py
# @Description :
import yaml
from service.rag_service import RAGService
from ui.app import GradioApp
from utils.logger import setup_logger


def main():
    setup_logger()

    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    rag_service = RAGService(config)
    app = GradioApp(rag_service)
    app.launch()


if __name__ == "__main__":
    main()