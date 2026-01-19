#!/usr/bin/env python3
"""
EIA Generator - Command Line Interface

Usage:
    python -m src.main --project "Solar Plant" --location "Ninh Thuan" --type energy_solar
    python -m src.main --interactive
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import ProjectInput, ProjectType, EIAConfig
from .orchestrator import EIAOrchestrator
from .generators.docx_generator import DocxGenerator

console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║           🌍 EIA GENERATOR - BÁO CÁO ĐTM                  ║
    ║         Environmental Impact Assessment System            ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold green")


def get_project_interactive() -> ProjectInput:
    """Get project info interactively."""
    console.print("\n📝 Nhập thông tin dự án:\n", style="bold")
    
    name = console.input("[bold]Tên dự án:[/bold] ")
    
    console.print("\nLoại dự án:")
    console.print("  1. Điện mặt trời")
    console.print("  2. Điện gió")
    console.print("  3. Sản xuất công nghiệp")
    console.print("  4. Đường giao thông")
    console.print("  5. Khu đô thị")
    
    type_map = {
        "1": ProjectType.ENERGY_SOLAR,
        "2": ProjectType.ENERGY_WIND,
        "3": ProjectType.INDUSTRIAL_MANUFACTURING,
        "4": ProjectType.INFRA_ROAD,
        "5": ProjectType.URBAN_RESIDENTIAL,
    }
    
    type_choice = console.input("\n[bold]Chọn loại (1-5):[/bold] ")
    project_type = type_map.get(type_choice, ProjectType.INDUSTRIAL_MANUFACTURING)
    
    location = console.input("[bold]Địa điểm:[/bold] ")
    area = float(console.input("[bold]Diện tích (ha):[/bold] ") or "50")
    capacity = console.input("[bold]Công suất:[/bold] ")
    investment = float(console.input("[bold]Vốn đầu tư (triệu USD):[/bold] ") or "10")
    
    return ProjectInput(
        name=name,
        type=project_type,
        location=location,
        area_hectares=area,
        capacity=capacity,
        investment_usd=investment * 1_000_000,
    )


async def generate_report(project: ProjectInput, output: str) -> None:
    """Generate EIA report."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Đang tạo báo cáo...", total=None)
        
        config = EIAConfig()
        orchestrator = EIAOrchestrator(config)
        
        progress.update(task, description="🔍 Nghiên cứu quy định...")
        report = await orchestrator.generate(project)
        
        progress.update(task, description="📄 Tạo file DOCX...")
        
        # Generate DOCX
        generator = DocxGenerator()
        output_path = generator.generate(report, output)
        
        progress.update(task, description="✅ Hoàn thành!")
    
    # Print results
    console.print("\n")
    
    table = Table(title="📊 Kết quả tạo báo cáo")
    table.add_column("Chỉ tiêu", style="cyan")
    table.add_column("Giá trị", style="green")
    
    table.add_row("Dự án", project.name)
    table.add_row("Điểm đánh giá", f"{report.compliance_score:.1f}/100")
    table.add_row("Số chương", str(len(report.sections)))
    table.add_row("File xuất", output_path)
    
    console.print(table)
    
    console.print(
        Panel(
            f"✅ Báo cáo đã được lưu tại: [bold]{output_path}[/bold]",
            title="Thành công",
            border_style="green",
        )
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EIA Generator - Tạo Báo cáo Đánh giá Tác động Môi trường",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="Tên dự án",
    )
    
    parser.add_argument(
        "--location", "-l",
        type=str,
        help="Địa điểm dự án",
    )
    
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="industrial_manufacturing",
        help="Loại dự án (energy_solar, energy_wind, industrial_manufacturing, ...)",
    )
    
    parser.add_argument(
        "--area",
        type=float,
        default=50,
        help="Diện tích (ha)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/eia_report.docx",
        help="File đầu ra",
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Chế độ tương tác",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị chi tiết",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    print_banner()
    
    if args.interactive:
        project = get_project_interactive()
    elif args.project and args.location:
        try:
            project_type = ProjectType(args.type)
        except ValueError:
            project_type = ProjectType.INDUSTRIAL_MANUFACTURING
        
        project = ProjectInput(
            name=args.project,
            type=project_type,
            location=args.location,
            area_hectares=args.area,
        )
    else:
        console.print(
            "[red]Lỗi: Cần nhập --project và --location, hoặc sử dụng --interactive[/red]"
        )
        sys.exit(1)
    
    # Confirm
    console.print(f"\n📋 Dự án: [bold]{project.name}[/bold]")
    console.print(f"📍 Địa điểm: {project.location}")
    console.print(f"🏭 Loại: {project.type.value}")
    console.print(f"📐 Diện tích: {project.area_hectares} ha\n")
    
    confirm = console.input("Tiếp tục tạo báo cáo? (y/n): ")
    if confirm.lower() != 'y':
        console.print("Đã hủy.")
        sys.exit(0)
    
    # Generate
    try:
        asyncio.run(generate_report(project, args.output))
    except KeyboardInterrupt:
        console.print("\n[yellow]Đã hủy.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Lỗi: {e}[/red]")
        if args.verbose:
            raise


if __name__ == "__main__":
    main()
