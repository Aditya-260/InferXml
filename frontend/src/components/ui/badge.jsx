import React from "react";
import { clsx } from "clsx";

function Badge({ className, variant = "default", ...props }) {
  const variants = {
    default: "badge-info",
    success: "badge-success",
    warning: "badge-warning",
    destructive: "badge-error",
    secondary: "bg-gray-800 text-gray-100 border border-gray-700",
    outline: "border border-gray-600 text-white",
  };

  return (
    <div
      className={clsx(
        "badge",
        variants[variant] || variants.default,
        className
      )}
      {...props}
    />
  );
}

export { Badge };
