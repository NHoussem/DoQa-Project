import React from "react";
import { useLocation } from "react-router-dom";
import { Logo } from "../Logo";
import { useNavigate } from "react-router-dom";
import { faArrowLeft } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

export const Navbar = ({ className }) => {
  const navigate = useNavigate();

  const handleReturnClick = () => {
    navigate("/");
  };
  const location = useLocation();
  const isAskPage = location.pathname.split("/").includes("ask");
  return (
    <div
      className={`flex flex-row w-full py-3 items-start justify-between relative bg-white shadow-[0px_4px_20px_#47556914] ${className}`}
    >
      {isAskPage && (
        <div className="flex items-centerjustify-center pl-4">
          <button
            onClick={handleReturnClick}
            className="bg-white rounded-3xl flex-col justify-center items-center  w-12 h-12 hover:bg-slate-100 "
          >
            <FontAwesomeIcon
              icon={faArrowLeft}
              className="text-[#0284C7] text-2xl "
              alt="Return"
            />
          </button>
        </div>
      )}

      <div className="flex w-full items-center justify-center pb-2 ">
        <Logo
          className="!h-[44.4px] !w-[130px] !relative"
          divClassName="!text-[25.6px] !left-[45px] !top-[2px]"
          groupClassName="!h-[36px] !left-[4px] !w-[126px] !top-[4px]"
          groupClassNameOverride="!h-[36px] !w-[38px]"
          overlapGroupClassName="!h-[36px]"
          rectangleClassName="!h-[28px] !rounded-[28px_28px_0px_28px] !w-[28px] !top-[8px]"
          rectangleClassNameOverride="!h-[28px] !rounded-[28px_28px_0px_28px] !left-[10px] !w-[28px]"
        />
      </div>
    </div>
  );
};
