import pandas as pd
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime


class AttendanceAnalyzer:
    """Модуль для анализа посещаемости студентов из CSV файлов"""

    def __init__(self, threshold_percent: float = 75.0):
        """
        Инициализация анализатора

        Args:
            threshold_percent: порог посещаемости (%), ниже которого студент считается проблемным
        """
        self.threshold_percent = threshold_percent
        self.attendance_data = []
        self.processed_files = []

    def load_csv_files(self, folder_path: str) -> List[str]:
        """
        Загрузка всех CSV файлов из указанной папки

        Args:
            folder_path: путь к папке с CSV файлами

        Returns:
            список путей к найденным CSV файлам
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Папка {folder_path} не найдена")

        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"В папке {folder_path} не найдено CSV файлов")

        return [str(f) for f in csv_files]

    def parse_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Парсинг отдельного CSV файла с данными посещаемости

        Ожидаемый формат CSV:
        - Столбцы: ФИО, Группа, Дисциплина, Дата1, Дата2, ..., ДатаN
        - Значения посещаемости: 1 (присутствовал), 0 (отсутствовал), пустая ячейка (не учитывается)

        Args:
            file_path: путь к CSV файлу

        Returns:
            DataFrame с обработанными данными
        """
        try:
            # Читаем CSV с автоопределением разделителя
            df = pd.read_csv(file_path, encoding='utf-8')

            # Если не получилось с utf-8, пробуем cp1251
            if df.empty or df.columns.size < 4:
                df = pd.read_csv(file_path, encoding='cp1251', sep=';')

            # Стандартизируем названия столбцов
            expected_cols = ['фио', 'группа', 'дисциплина']
            df.columns = df.columns.str.lower().str.strip()

            # Проверяем наличие столбцов
            for col in expected_cols:
                if col not in df.columns:
                    # находим похожие столбцы
                    similar_cols = [c for c in df.columns if col in c or c in col]
                    if similar_cols:
                        df = df.rename(columns={similar_cols[0]: col})
                    else:
                        raise ValueError(f"В файле {file_path} не найден столбец '{col}'")

            # определяем столбцы с датами (все кроме первых трех)
            date_columns = [col for col in df.columns if col not in expected_cols]

            if not date_columns:
                raise ValueError(f"В файле {file_path} не найдены столбцы с датами")

            # очищаем данные
            df = df.dropna(subset=['фио', 'группа', 'дисциплина'])
            df['фио'] = df['фио'].str.strip()
            df['группа'] = df['группа'].str.strip()
            df['дисциплина'] = df['дисциплина'].str.strip()

            # преобразуем данные посещаемости в числовой формат
            for col in date_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['source_file'] = Path(file_path).name

            return df

        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {str(e)}")
            return pd.DataFrame()

    def calculate_attendance_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет статистики посещаемости для каждого студента

        Args:
            df: DataFrame с данными посещаемости

        Returns:
            DataFrame со статистикой по каждому студенту
        """
        stats = []

        for _, row in df.iterrows():
            # получаем столбцы с датами
            date_columns = [col for col in df.columns
                            if col not in ['фио', 'группа', 'дисциплина', 'source_file']]

            # Считаем посещаемость (только для непустых значений)
            attendance_values = [row[col] for col in date_columns if pd.notna(row[col])]

            if attendance_values:
                total_classes = len(attendance_values)
                attended_classes = sum(attendance_values)
                attendance_percent = (attended_classes / total_classes) * 100
            else:
                total_classes = 0
                attended_classes = 0
                attendance_percent = 0

            stats.append({
                'фио': row['фио'],
                'группа': row['группа'],
                'дисциплина': row['дисциплина'],
                'всего_занятий': total_classes,
                'посещено': attended_classes,
                'пропущено': total_classes - attended_classes,
                'процент_посещаемости': round(attendance_percent, 2),
                'проблемная_посещаемость': attendance_percent < self.threshold_percent,
                'источник': row['source_file']
            })

        return pd.DataFrame(stats)

    def process_all_files(self, folder_path: str) -> pd.DataFrame:
        """
        Обработка всех CSV файлов из папки

        Args:
            folder_path: путь к папке с CSV файлами

        Returns:
            DataFrame с общей статистикой
        """
        csv_files = self.load_csv_files(folder_path)
        all_stats = []

        print(f"Найдено {len(csv_files)} CSV файлов для обработки:")

        for file_path in csv_files:
            print(f"Обрабатывается: {Path(file_path).name}")

            df = self.parse_csv_file(file_path)
            if not df.empty:
                stats = self.calculate_attendance_stats(df)
                all_stats.append(stats)
                self.processed_files.append(file_path)
            else:
                print(f"Файл {file_path} пропущен из-за ошибок")

        if all_stats:
            return pd.concat(all_stats, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_problem_students(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Получение списка студентов с проблемной посещаемостью

        Args:
            stats_df: DataFrame со статистикой

        Returns:
            DataFrame с проблемными студентами
        """
        problem_students = stats_df[stats_df['проблемная_посещаемость']].copy()
        return problem_students.sort_values('процент_посещаемости')

    def get_group_statistics(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет статистики по группам

        Args:
            stats_df: DataFrame со статистикой

        Returns:
            DataFrame со статистикой по группам
        """
        group_stats = stats_df.groupby('группа').agg({
            'процент_посещаемости': ['mean', 'min', 'max', 'count'],
            'проблемная_посещаемость': 'sum'
        }).round(2)

        group_stats.columns = ['средняя_посещаемость', 'мин_посещаемость',
                               'макс_посещаемость', 'количество_студентов',
                               'проблемных_студентов']

        group_stats = group_stats.reset_index()
        group_stats['процент_проблемных'] = round(
            (group_stats['проблемных_студентов'] / group_stats['количество_студентов']) * 100, 2
        )

        return group_stats.sort_values('средняя_посещаемость')

    def get_discipline_statistics(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет статистики по дисциплинам

        Args:
            stats_df: DataFrame со статистикой

        Returns:
            DataFrame со статистикой по дисциплинам
        """
        discipline_stats = stats_df.groupby('дисциплина').agg({
            'процент_посещаемости': ['mean', 'min', 'max', 'count'],
            'проблемная_посещаемость': 'sum'
        }).round(2)

        discipline_stats.columns = ['средняя_посещаемость', 'мин_посещаемость',
                                    'макс_посещаемость', 'количество_студентов',
                                    'проблемных_студентов']

        discipline_stats = discipline_stats.reset_index()
        discipline_stats['процент_проблемных'] = round(
            (discipline_stats['проблемных_студентов'] / discipline_stats['количество_студентов']) * 100, 2
        )

        return discipline_stats.sort_values('средняя_посещаемость')

    def generate_report(self, stats_df: pd.DataFrame) -> str:
        """
        Генерация текстового отчета

        Args:
            stats_df: DataFrame со статистикой

        Returns:
            строка с отчетом
        """
        if stats_df.empty:
            return "Нет данных для анализа"

        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ ПО АНАЛИЗУ ПОСЕЩАЕМОСТИ СТУДЕНТОВ")
        report.append("=" * 80)
        report.append(f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        report.append(f"Порог проблемной посещаемости: {self.threshold_percent}%")
        report.append(f"Обработано файлов: {len(self.processed_files)}")
        report.append("")

        # Общая статистика
        total_students = len(stats_df)
        problem_students = len(stats_df[stats_df['проблемная_посещаемость']])
        avg_attendance = stats_df['процент_посещаемости'].mean()

        report.append("ОБЩАЯ СТАТИСТИКА:")
        report.append(f"• Всего студентов: {total_students}")
        report.append(f"• Студентов с проблемной посещаемостью: {problem_students}")
        report.append(f"• Процент проблемных студентов: {round((problem_students / total_students) * 100, 2)}%")
        report.append(f"• Средняя посещаемость: {round(avg_attendance, 2)}%")
        report.append("")

        # Самая проблемная группа
        group_stats = self.get_group_statistics(stats_df)
        worst_group = group_stats.iloc[0]

        report.append("САМАЯ ПРОБЛЕМНАЯ ГРУППА:")
        report.append(f"• Группа: {worst_group['группа']}")
        report.append(f"• Средняя посещаемость: {worst_group['средняя_посещаемость']}%")
        report.append(
            f"• Проблемных студентов: {worst_group['проблемных_студентов']} из {worst_group['количество_студентов']} ({worst_group['процент_проблемных']}%)")
        report.append("")

        # Топ-10 проблемных студентов
        problem_students_df = self.get_problem_students(stats_df)
        report.append("ТОП-10 СТУДЕНТОВ С НАИХУДШЕЙ ПОСЕЩАЕМОСТЬЮ:")
        for i, (_, student) in enumerate(problem_students_df.head(10).iterrows(), 1):
            report.append(f"{i:2}. {student['фио']} (гр. {student['группа']}) - {student['процент_посещаемости']}% "
                          f"({student['посещено']}/{student['всего_занятий']})")

        return "\n".join(report)

    def save_detailed_results(self, stats_df: pd.DataFrame, output_folder: str = "results"):
        """
        Сохранение детальных результатов в Excel файлы

        Args:
            stats_df: DataFrame со статистикой
            output_folder: папка для сохранения результатов
        """
        if stats_df.empty:
            print("Нет данных для сохранения")
            return

        # Создаем папку для результатов
        Path(output_folder).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Общая статистика
        excel_path = Path(output_folder) / f"attendance_analysis_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            stats_df.to_excel(writer, sheet_name='Все студенты', index=False)

            problem_students = self.get_problem_students(stats_df)
            problem_students.to_excel(writer, sheet_name='Проблемные студенты', index=False)

            group_stats = self.get_group_statistics(stats_df)
            group_stats.to_excel(writer, sheet_name='Статистика по группам', index=False)

            discipline_stats = self.get_discipline_statistics(stats_df)
            discipline_stats.to_excel(writer, sheet_name='Статистика по предметам', index=False)

        print(f"Детальные результаты сохранены в: {excel_path}")

        # Текстовый отчет
        report_path = Path(output_folder) / f"attendance_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report(stats_df))

        print(f"Текстовый отчет сохранен в: {report_path}")


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Анализ посещаемости студентов из CSV файлов')
    parser.add_argument('folder', help='Путь к папке с CSV файлами')
    parser.add_argument('--threshold', type=float, default=75.0,
                        help='Порог посещаемости в процентах (по умолчанию 75)')
    parser.add_argument('--output', default='results',
                        help='Папка для сохранения результатов (по умолчанию "results")')

    args = parser.parse_args()

    try:
        # создаем анализатор
        analyzer = AttendanceAnalyzer(threshold_percent=args.threshold)

        # обрабатываем файлы
        print("Начинаем обработку файлов...")
        stats_df = analyzer.process_all_files(args.folder)

        if stats_df.empty:
            print("Не удалось обработать ни одного файла")
            return

        # выводим отчет в консоль
        print("\n" + analyzer.generate_report(stats_df))

        # сохраняем детальные результаты
        analyzer.save_detailed_results(stats_df, args.output)

        print(f"\nОбработка завершена успешно!")

    except Exception as e:
        print(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    main()

