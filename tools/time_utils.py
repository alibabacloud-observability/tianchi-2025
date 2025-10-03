"""Time utilities for parsing time expressions and ranges

Migrated from mcp_server_aliyun_observability.toolkits.paas.time_utils
with enhanced error handling and Pydantic integration.
"""

import re
import time
from datetime import datetime
from typing import Tuple, Union
from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    """Time range model for queries"""

    from_timestamp: int = Field(...,
                                description="Start time as Unix timestamp (seconds)")
    to_timestamp: int = Field(...,
                              description="End time as Unix timestamp (seconds)")
    from_time_str: str = Field(...,
                               description="Original from_time expression")
    to_time_str: str = Field(..., description="Original to_time expression")


class TimeRangeParser:
    """时间范围解析工具类

    支持两种时间格式的解析：
    1. Unix时间戳（整数）
    2. 相对时间表达式（如 "now-1h", "now-30m", "now-1d"）
    """

    @staticmethod
    def parse_time_expression(time_expr: Union[str, int]) -> int:
        """解析时间表达式为Unix时间戳（秒）

        Args:
            time_expr: 时间表达式，支持：
                - Unix时间戳（整数，秒或毫秒）
                - 相对时间表达式：now-1h, now-30m, now-1d, now-7d

        Returns:
            Unix时间戳（秒）

        Examples:
            parse_time_expression(1640995200) -> 1640995200 (秒时间戳)
            parse_time_expression(1640995200000) -> 1640995200 (毫秒转秒)
            parse_time_expression("now-1h") -> 当前时间-1小时的时间戳
            parse_time_expression("now-30m") -> 当前时间-30分钟的时间戳
        """
        # 如果是整数，需要判断是秒还是毫秒时间戳
        if isinstance(time_expr, int):
            return TimeRangeParser._normalize_timestamp(time_expr)

        if isinstance(time_expr, str) and time_expr.isdigit():
            return TimeRangeParser._normalize_timestamp(int(time_expr))

        # 解析相对时间表达式
        if isinstance(time_expr, str) and time_expr.startswith("now"):
            return TimeRangeParser._parse_relative_time(time_expr)

        # 如果都不匹配，尝试直接转换为整数
        try:
            timestamp = int(float(time_expr))
            return TimeRangeParser._normalize_timestamp(timestamp)
        except (ValueError, TypeError):
            raise ValueError(f"不支持的时间格式: {time_expr}")

    @staticmethod
    def _normalize_timestamp(timestamp: int) -> int:
        """标准化时间戳为秒级

        自动判断输入的时间戳是秒还是毫秒，并转换为秒级时间戳

        Args:
            timestamp: 时间戳（秒或毫秒）

        Returns:
            秒级时间戳
        """
        # 判断是否为毫秒时间戳
        if timestamp > 1e11:  # 5138年11月16日
            return timestamp // 1000
        else:
            return timestamp

    @staticmethod
    def _parse_relative_time(time_expr: str) -> int:
        """解析相对时间表达式

        Args:
            time_expr: 相对时间表达式，如 "now-1h", "now-30m"

        Returns:
            Unix时间戳（秒）
        """
        now = int(time.time())

        # 如果只是 "now"
        if time_expr.strip().lower() == "now":
            return now

        # 匹配模式: now-{数字}{单位}
        pattern = r'^now([+-])(\d+)([smhd])$'
        match = re.match(pattern, time_expr.strip().lower())

        if not match:
            raise ValueError(
                f"无效的相对时间格式: {time_expr}. 支持格式: now, now-1h, now-30m, now-1d")

        operator, amount_str, unit = match.groups()
        amount = int(amount_str)

        # 计算时间偏移（秒）
        unit_multipliers = {
            's': 1,          # 秒
            'm': 60,         # 分钟
            'h': 3600,       # 小时
            'd': 86400,      # 天
        }

        if unit not in unit_multipliers:
            raise ValueError(f"不支持的时间单位: {unit}. 支持单位: s, m, h, d")

        offset_seconds = amount * unit_multipliers[unit]

        # 根据操作符计算最终时间
        if operator == '-':
            return now - offset_seconds
        else:  # operator == '+'
            return now + offset_seconds

    @staticmethod
    def parse_time_range(
            from_time: Union[str, int],
            to_time: Union[str, int]) -> TimeRange:
        """解析时间范围

        Args:
            from_time: 开始时间表达式
            to_time: 结束时间表达式

        Returns:
            TimeRange对象包含解析后的时间戳和原始表达式

        Examples:
            parse_time_range("now-1h", "now") -> TimeRange with timestamps
            parse_time_range(1640995200, 1640998800) -> TimeRange with timestamps
        """
        from_timestamp = TimeRangeParser.parse_time_expression(from_time)
        to_timestamp = TimeRangeParser.parse_time_expression(to_time)

        # 确保时间范围有效
        if from_timestamp >= to_timestamp:
            raise ValueError(f"开始时间({from_timestamp})必须小于结束时间({to_timestamp})")

        return TimeRange(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            from_time_str=str(from_time),
            to_time_str=str(to_time)
        )

    @staticmethod
    def from_string(time_range_str: str) -> TimeRange:
        """解析组合时间范围格式

        支持格式: "2025-09-12 15:42:09 ~ 2025-09-12 15:47:29"

        Args:
            time_range_str: 时间范围字符串，如 "2025-09-12 15:42:09 ~ 2025-09-12 15:47:29"

        Returns:
            TimeRange对象，包含解析后的时间戳

        Raises:
            ValueError: 如果时间格式无效
        """
        # 匹配格式: YYYY-MM-DD HH:MM:SS ~ YYYY-MM-DD HH:MM:SS
        pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ~ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$'
        match = re.match(pattern, time_range_str.strip())

        if not match:
            raise ValueError(
                f"Invalid time range format: {time_range_str}. Expected format: 'YYYY-MM-DD HH:MM:SS ~ YYYY-MM-DD HH:MM:SS'"
            )

        from_time_str = match.group(1)
        to_time_str = match.group(2)

        try:
            # 解析开始时间
            from_dt = datetime.strptime(from_time_str, '%Y-%m-%d %H:%M:%S')
            from_timestamp = int(from_dt.timestamp())

            # 解析结束时间
            to_dt = datetime.strptime(to_time_str, '%Y-%m-%d %H:%M:%S')
            to_timestamp = int(to_dt.timestamp())

            # 确保时间范围有效
            if from_timestamp >= to_timestamp:
                raise ValueError(
                    f"开始时间({from_timestamp})必须小于结束时间({to_timestamp})"
                )

            return TimeRange(
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                from_time_str=from_time_str,
                to_time_str=to_time_str
            )

        except ValueError as e:
            raise ValueError(
                f"Failed to parse datetime from {time_range_str}: {e}"
            )
